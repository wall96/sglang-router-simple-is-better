//! LMETRIC-style cache-aware load balancing policy.
//!
//! This policy follows the multiplicative score from "Simple is Better:
//! Multiplication May Be All You Need for LLM Request Scheduling":
//!
//!     score_i = P_token_i * (BS_i + 1)
//!
//! Following the paper (§5.1), the KV$-aware indicator P_token has TWO parts:
//!
//!     P_token_i = queued_prefill_tokens_i + new_tokens_i(req)
//!
//! where `new_tokens_i(req)` is this request's uncached prefill work on worker
//! i, and `queued_prefill_tokens_i` is the backlog of not-yet-drained prefill
//! tokens from other in-flight requests already routed to i. The queued term is
//! what gives P_token its load-balancing power over a plain hit-ratio: the
//! router steers away from instances whose prefill queue is piling up, even if
//! their cache-hit rate is high.
//!
//! In this router the cache index is the existing approximate text radix tree,
//! so token counts are approximated with input *character* counts, and
//! `queued_prefill_tokens_i` is tracked per-worker (incremented on admission,
//! decremented via the request's `WorkerLoadGuard` on completion). NOTE: unlike
//! the paper — which drains the queued term at *prefill* completion via engine
//! step-level SSE — this HTTP router has no prefill-complete signal, so the
//! term is drained at *request* completion (i.e. it lingers through the decode
//! phase). This slightly over-counts prefill backlog for long-decode requests
//! but preserves the paper's load-balancing intent.
//!
//! The routing target key is always `worker.url()`: non-DP-aware deployments
//! route at worker/node granularity, while DP-aware deployments naturally route
//! at `base_url@dp_rank` granularity.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use smg_mesh::{tree_ops::TreeOperation, OptionalMeshSyncManager};
use tracing::{debug, warn};

use super::{
    cache_aware::tree_key_for_worker, get_healthy_worker_indices, tree::Tree, utils::PeriodicTask,
    CacheAwareConfig, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::{Worker, UNKNOWN_MODEL_ID};

#[derive(Debug)]
pub struct LMetricPolicy {
    trees: Arc<DashMap<String, Arc<Tree>>>,
    mesh_sync: OptionalMeshSyncManager,
    _eviction_task: Option<PeriodicTask>,
}

impl LMetricPolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "LMetric eviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        let tree_key = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);
                        debug!(
                            "LMETRIC cache eviction completed for {}, max_size: {}",
                            tree_key, max_tree_size
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            trees,
            mesh_sync: None,
            _eviction_task: eviction_task,
        }
    }

    pub fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        self.mesh_sync = mesh_sync.clone();
        if mesh_sync.is_some() {
            self.restore_tree_state_from_mesh();
        }
    }

    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        let mut grouped: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            grouped
                .entry(tree_key_for_worker(worker.as_ref()))
                .or_default()
                .push(worker);
        }

        for (tree_key, pool_workers) in grouped {
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()));
            for worker in pool_workers {
                tree.insert("", worker.url());
            }
        }
    }

    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        let tree = self
            .trees
            .entry(tree_key)
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        if let Some(tree) = self.trees.get(&tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    pub fn remove_worker_by_url(&self, url: &str) {
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    fn restore_tree_state_from_mesh(&self) {
        if let Some(ref mesh_sync) = self.mesh_sync {
            for tree_ref in self.trees.iter() {
                let tree_key = tree_ref.key();
                if let Some(tree_state) = mesh_sync.get_tree_state(tree_key) {
                    debug!(
                        "Restoring LMETRIC tree state for {} with {} operations",
                        tree_key,
                        tree_state.operations.len()
                    );

                    let tree = tree_ref.value();
                    for operation in &tree_state.operations {
                        match operation {
                            TreeOperation::Insert(insert_op) => {
                                tree.insert(&insert_op.text, &insert_op.tenant);
                            }
                            TreeOperation::Remove(remove_op) => {
                                tree.remove_tenant(&remove_op.tenant);
                            }
                        }
                    }
                }
            }
        }
    }

    fn normalize_mesh_model_id(tree_key: &str) -> &str {
        if tree_key.is_empty() {
            UNKNOWN_MODEL_ID
        } else {
            tree_key
        }
    }

    pub fn apply_remote_tree_operation(&self, mesh_key: &str, operation: &TreeOperation) {
        let tree_key = Self::normalize_mesh_model_id(mesh_key);

        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));

        match operation {
            TreeOperation::Insert(insert_op) => {
                tree.insert(&insert_op.text, &insert_op.tenant);
                debug!(
                    "Applied remote LMETRIC tree insert: key={}, text={}, tenant={}",
                    mesh_key, insert_op.text, insert_op.tenant
                );
            }
            TreeOperation::Remove(remove_op) => {
                tree.remove_tenant(&remove_op.tenant);
                debug!(
                    "Applied remote LMETRIC tree remove: key={}, tenant={}",
                    mesh_key, remove_op.tenant
                );
            }
        }
    }

    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let tree_key = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "LMETRIC cache eviction for {}, max_size: {}",
                tree_key, max_size
            );
        }
    }

    fn update_tree(&self, tree: &Tree, tree_key: &str, text: &str, worker_url: &str) {
        tree.insert(text, worker_url);

        if let Some(ref mesh_sync) = self.mesh_sync {
            use smg_mesh::tree_ops::TreeInsertOp;
            let op = TreeOperation::Insert(TreeInsertOp {
                text: text.to_string(),
                tenant: worker_url.to_string(),
            });
            let mesh_key = Self::normalize_mesh_model_id(tree_key);
            if let Err(e) = mesh_sync.sync_tree_operation(mesh_key.to_string(), op) {
                warn!(
                    "Failed to sync LMETRIC tree insert operation to mesh: {}",
                    e
                );
            }
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for LMetricPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let request_text = info.request_text;
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        let Some(text) = request_text else {
            let selected_idx = healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()?;
            // Balance the WorkerLoadGuard pop: every returned selection pushes
            // exactly one backlog entry (0 here — no text to score for prefill).
            workers[selected_idx].push_queued_prefill_tokens(0);
            workers[selected_idx].increment_processed();
            return Some(selected_idx);
        };

        let pivot = workers[healthy_indices[0]].as_ref();
        let tree_key = tree_key_for_worker(pivot);
        let input_char_count = text.chars().count();

        let tree = self.trees.get(&tree_key).map(|entry| entry.value().clone());
        let selected_idx = if let Some(ref tree) = tree {
            healthy_indices
                .iter()
                .min_by_key(|&&idx| {
                    let matched_chars = tree.prefix_match_tenant_count(text, workers[idx].url());
                    let new_tokens = input_char_count.saturating_sub(matched_chars);
                    // P-token = queued prefill backlog already at this instance
                    // + this request's new (uncached) prefill tokens. The queued
                    // term is what carries the prefill-side load-balancing signal
                    // (paper §5.1: P-token beats hit-ratio precisely because it
                    // also reflects each instance's queued prefill work).
                    let p_tokens = workers[idx]
                        .queued_prefill_tokens()
                        .saturating_add(new_tokens);
                    // (bs + 1): keep idle workers (load 0) distinguishable by
                    // P-token alone, matching the paper's (BS + 1) term.
                    let batch_size = workers[idx].load().saturating_add(1);
                    p_tokens.saturating_mul(batch_size)
                })
                .copied()
        } else {
            warn!(
                "lmetric: no tree found for key '{}', falling back to min-load worker selection",
                tree_key
            );
            healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()
        }?;

        if let Some(tree) = tree {
            // Record this request's new (uncached) prefill tokens as queued
            // backlog on the chosen worker. Must be computed BEFORE update_tree
            // inserts the text (which would make the prefix fully match → 0).
            let matched_chars =
                tree.prefix_match_tenant_count(text, workers[selected_idx].url());
            let new_tokens = input_char_count.saturating_sub(matched_chars);
            workers[selected_idx].push_queued_prefill_tokens(new_tokens);
            self.update_tree(tree.as_ref(), &tree_key, text, workers[selected_idx].url());
        } else {
            // No tree existed yet → nothing cached, so all input tokens are new.
            workers[selected_idx].push_queued_prefill_tokens(input_char_count);
            let tree = self
                .trees
                .entry(tree_key.clone())
                .or_insert_with(|| Arc::new(Tree::new()));
            self.update_tree(
                tree.value().as_ref(),
                &tree_key,
                text,
                workers[selected_idx].url(),
            );
        }

        workers[selected_idx].increment_processed();
        Some(selected_idx)
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        if !success {
            tracing::debug!(
                "LMETRIC request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn name(&self) -> &'static str {
        "lmetric"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        LMetricPolicy::set_mesh_sync(self, mesh_sync);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for LMetricPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType};

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .build(),
        )
    }

    #[tokio::test]
    async fn test_lmetric_prefers_cached_worker_when_load_is_close() {
        let policy = LMetricPolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers = vec![worker("http://w1:8000"), worker("http://w2:8000")];
        policy.init_workers(&workers);

        let idx1 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("shared prefix A"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Simulate request 1's prefill draining (WorkerLoadGuard pops on
        // completion); it stays "running" in decode via the load bump below.
        workers[idx1].pop_queued_prefill_tokens();
        workers[idx1].increment_load();
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("shared prefix A"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(idx1, idx2);
    }

    #[tokio::test]
    async fn test_lmetric_trades_cache_for_load() {
        let policy = LMetricPolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers = vec![worker("http://w1:8000"), worker("http://w2:8000")];
        policy.init_workers(&workers);

        let cached_idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("abcdefghij"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let other_idx = 1 - cached_idx;

        // Request 1 completes: its queued prefill drains (as WorkerLoadGuard
        // would do), leaving only the warm cache behind.
        workers[cached_idx].pop_queued_prefill_tokens();
        for _ in 0..20 {
            workers[cached_idx].increment_load();
        }

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("abcdefghijk"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(selected, other_idx);
    }

    #[tokio::test]
    async fn test_lmetric_without_request_text_falls_back_to_min_load() {
        let policy = LMetricPolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers = vec![worker("http://w1:8000"), worker("http://w2:8000")];
        policy.init_workers(&workers);

        workers[0].increment_load();
        workers[0].increment_load();

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: None,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(selected, 1);
    }

    #[tokio::test]
    async fn test_lmetric_uses_worker_url_as_dp_rank_target_key() {
        let policy = LMetricPolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                DPAwareWorkerBuilder::new("http://node:8000", 0, 2)
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                DPAwareWorkerBuilder::new("http://node:8000", 1, 2)
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("rank-local-prefix"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Request 1 completes on the chosen rank (drain its queued prefill).
        workers[idx].pop_queued_prefill_tokens();

        let second = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("rank-local-prefix"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(idx, second);
        assert!(workers[idx].url().ends_with("@0") || workers[idx].url().ends_with("@1"));
    }

    /// The queued-prefill backlog (paper §5.1) must steer new requests away
    /// from an instance that already has heavy *pending* prefill work, even
    /// when raw request counts (load) are equal. This is exactly the signal
    /// the old implementation (new-tokens only) was missing.
    #[tokio::test]
    async fn test_lmetric_queued_prefill_backlog_balances_load() {
        let policy = LMetricPolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers = vec![worker("http://w1:8000"), worker("http://w2:8000")];
        policy.init_workers(&workers);

        // Admit a request with a large, uncached prompt to w0 and leave it
        // in-flight (no pop) → w0 carries a big queued-prefill backlog.
        let big_prompt = "x".repeat(500);
        let busy_idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(big_prompt.as_str()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(workers[busy_idx].queued_prefill_tokens() >= 500);

        // A new, unrelated (also uncached) request: both workers have equal
        // load (0) and no cache hit, so a load-only or new-tokens-only score
        // would tie/pick w0. The queued backlog must push it to the other.
        let other_idx = 1 - busy_idx;
        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("a fresh unrelated prompt"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(selected, other_idx);
    }
}
