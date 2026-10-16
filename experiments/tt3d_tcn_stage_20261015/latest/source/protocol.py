"""One fixed candidate-filter family. No new OOF training or threshold search."""
SEEDS=(1729,2718,31415)
VARIANTS=('calibrated_event','fps_context')
POLICIES=('calibrated_control','fps_frame_agreement','fps_time_agreement')
BASE_DECODER={'threshold':.5,'minimum_separation_s':.12}
CONFIGS=(BASE_DECODER,)
FIXED={'epochs':120,'seeds':list(SEEDS),'channels':32,'dilations':[1,2,4,8],'dropout':.1,
 'learning_rate':.002,'weight_decay':.01,'gradient_clip':1.,'padding_mask':True,
 'feature_widths':{'calibrated_event':51,'fps_context':56},
 'strong_threshold':.65,'fps_support_threshold':.65,'frame_support_radius':1,'time_support_radius_s':.12}
POLICY={'schema':1,'candidate_policies':list(POLICIES),
 'budget':{'new_inner_seed_fits':0,'maximum_final_fps_seed_fits':3,'reused_inner_seed_models':18,'reused_final_base_seed_models':3},
 'base':'Decode frozen calibrated scores at threshold0.5 and interval0.12s. Keep its event times and actual detected coordinates.',
 'weak_fallback':'Every decoded calibrated peak below0.65 is retained even if FPS branch has zero support.',
 'strong_check':'For a calibrated peak >=0.65, retain only when max FPS touch probability >=0.65 within the fixed support neighborhood.',
 'alignment_options':'Exactly two fixed options: +/-1 original frame, or +/-0.12s actual PTS (existing coordinate/decode scales). No support radius/threshold sweeps.',
 'no_new_events':'Filter original candidates only. No new peaks, no re-running NMS after filtering, no event retiming or coordinate replacement.',
 'selection':'No currently matched touch identity can be lost in complete or trusted scope. Overall and per-view0/45 recall/F1 cannot fall, overall FP cannot grow, 90 complete F1 cannot fall. At least one complete/trusted weak-view FP count must decrease.',
 'ranking':'Eligible complete F1, trusted F1, fewer complete FP, simpler frame support. Otherwise retain calibrated control.',
 'source_isolation':'Byte-identical previous three source/view-stratified inner folds and frozen OOF branch scores. Prior models trained on fitting sources only; no historical checkpoint initialization for new final FPS fits.',
 'post_selection_training':'If a gate passes, fit exactly three final FPS models from scratch on the same42 development videos. Reuse current fixed three final base models. No reselecting based on final training scores.',
 'no_specific_case_rules':'No video/frame identity checks, handwritten keep/remove lists, annotated event timing or hit vetoes in the policy.',
 'outer_reads_allowed':False,'public_or_outer_predictions':False,'independent_blind_test':False,
 'exposure':'Historically exposed development data and repeated same-inner development. Not an independent generalization test.',
 'runtime':'Only video plus fixed model assets; automatic geometry/trajectory/PTS. No manual labels, corners, ball boxes, source IDs or historical cache inputs.',
 'stopping':'One frozen policy family, two alignment options, maximum3 new fits. No new thresholds or policies after scores.',
 'annotation':'Existing complete/window event records and provenance reused. Check all earlier records before any new request.'}
