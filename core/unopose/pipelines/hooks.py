"""
Hook points for the runtime pipeline components (m2, verifier, ...).  Everything is OFF by default (`select` None, `before_batch` empty), so the original code paths are unchanged.

select        callable(idx, scores, pred_rs, pred_ts, model_pts) -> idx.  Called inside `compute_coarse_Rt_overlap` (core/unopose/utils/model_utils.py) with the pipeline's own choice `idx` (B,) among
              the `n_proposal2` coarse hypotheses; returns the index to use instead.  pred_rs (B,C,3,3) and pred_ts (B,C,1,3) are radius-normalised, template -> query.
before_batch  callables(data, inputs, start_idx, end_idx, data_loader).  Called by `inference_and_save_oneref_v1` (core/unopose/engine/oneref_inference_utils_v1.py) before each batch of
              instances goes through the model; used to hand per-instance context (query depth map, mask, query points) to `select`.
guide         callable(dis, pred_rs, k) -> idx (B,k).  Called inside `compute_coarse_Rt_overlap` instead of the plain "k proposals with the smallest residual" step: `dis` (B,n_proposal1) is the mean residual of each of the
              6000 sampled proposals, `pred_rs` (B,n_proposal1,3,3) their rotations; returns the indices of the k proposals that form the pool of hypotheses (the `gpool` component).
fine_atten    callable(atten, pts2, model_pts) -> atten.  Called at the start of `compute_fine_Rt_overlap` (core/unopose/utils/model_utils.py) with the fine stage's similarity matrix (B, N1+1, N2+1)
              before the dual-softmax; returns the matrix to use instead (RoMa adds its evidence here).
after_batch   callables(data, inputs, end_points, start_idx, end_idx).  Called by `inference_and_save_oneref_v1` right after the model has run on a batch; read-only use (the `diag` recorder).
state         values the model code leaves for the hooks (`roma`: RoMa's (target, cycle-error) maps of the instances of the current batch, set by the roma component when it is on): `radius` (B,), the object radius used to normalise the point clouds of the instances in the current batch; `model_pts` (B,Nm,3), the normalised
              reference points (stored by the selection hook, so only available when `m2` or `verifier` is on); `corr` (B,N1,N2) the dual-softmax correspondence weights of the coarse stage and `pts1` (B,N1,3) the sparse query points (stored before the selection hook is called).
"""
select = None
guide = None
fine_atten = None
before_batch = []
after_batch = []
state = {"radius": None, "model_pts": None, "corr": None, "pts1": None, "roma": None}
