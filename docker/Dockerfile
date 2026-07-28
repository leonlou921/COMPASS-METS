# syntax=docker.io/docker/dockerfile:1.12
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime@sha256:ac7c098a81512e719afa5d2d497f812d7db3498f340a4b819c69cb7b3b257126
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="BraTS MET N03 final inference"
LABEL org.opencontainers.image.description="Frozen N03_XF12_LCv3_ET_parent_supported candidate"
LABEL org.opencontainers.image.version="2026.07.27"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    nnUNet_results=/opt/n03/assets/nnUNet_results \
    nnUNet_raw=/opt/n03/empty/nnUNet_raw \
    nnUNet_preprocessed=/opt/n03/empty/nnUNet_preprocessed \
    PYTHONPATH=/opt/n03/app:/opt/n03/vendor/lcv1:/opt/n03/vendor/portfolio:/opt/n03/vendor/pipeline

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/n03

COPY docker/requirements.lock docker/constraints.lock /opt/n03/
RUN python -m pip install --no-cache-dir \
    --constraint /opt/n03/constraints.lock \
    --requirement /opt/n03/requirements.lock

COPY vendor/nnUNet /opt/n03/vendor/nnUNet
RUN python -m pip install --no-cache-dir --no-deps /opt/n03/vendor/nnUNet

COPY inference/vendor/pipeline /opt/n03/vendor/pipeline
COPY inference/vendor/lcv1 /opt/n03/vendor/lcv1
COPY inference/vendor/portfolio /opt/n03/vendor/portfolio
COPY inference/src/n03_docker /opt/n03/app/n03_docker
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/plans.json assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/dataset.json /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_0 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_0
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_1 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_1
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_2 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_2
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_3 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_3
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_4 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetXL30GBPlans__3d_fullres/fold_4
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/plans.json assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/dataset.json /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_1 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_1
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_2 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_2
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_3 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_3
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_4 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_4
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/plans.json assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/dataset.json /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_0 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_0
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_1 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_1
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_2 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_2
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_3 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_3
COPY assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_4 /opt/n03/assets/nnUNet_results/Dataset501_BraTS2025MET/nnUNetTrainer_ResEncM_DiceCEFocalTverskyFT__nnUNetResEncUNetMPlans__3d_fullres/fold_4
COPY assets/learned_models /opt/n03/assets/learned_models
COPY assets/provenance /opt/n03/assets/provenance
COPY provenance /opt/n03/provenance

RUN mkdir -p /opt/n03/empty/nnUNet_raw /opt/n03/empty/nnUNet_preprocessed \
    && python -c "import lightgbm, nnunetv2, torch; from n03_docker.postprocess import CANDIDATE_ID; assert CANDIDATE_ID == 'N03_XF12_LCv3_ET_parent_supported'; assert torch.__version__.startswith('2.4.1')"

ENTRYPOINT ["python", "-m", "n03_docker.cli"]
