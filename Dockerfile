# typestrict-serve: the HTTP runtime for typestrict.ai models, Jev's decision API on CPU. Published as
# ghcr.io/typestrict-lab/serve, linux/amd64 and linux/arm64.
#
# This image carries NO model. A model's image is this one plus one layer with its files at /model:
#
#   FROM ghcr.io/typestrict-lab/serve:0.1
#   COPY --chown=root:root model/ /model/      # decision_ai.json, tokenizer.json, onnx/model_8bit.onnx
#
# or mount them: docker run -p 8080:8080 -v "$PWD/model:/model:ro" ghcr.io/typestrict-lab/serve
#
# ONNX Runtime is linked statically (ort's prebuilt for *-unknown-linux-gnu), so the runtime image only needs glibc and
# libstdc++, both already in debian slim. The AWS Lambda Web Adapter rides along as a Lambda extension: the same image
# runs on Lambda unchanged, and does nothing anywhere else.

FROM rust:1.98-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
 && cp target/release/typestrict-serve /typestrict-serve

FROM debian:trixie-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0@sha256:17cfd08eff1dfea3f6a9a1e9c65fdac80aa4919b6085e746615530f43f57d2f1 \
     /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=build /typestrict-serve /usr/local/bin/typestrict-serve
RUN useradd --uid 10001 --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin serve \
 && mkdir -p /model
# PORT is also what the Lambda adapter forwards to; it waits for /healthz to answer 200 (503 while the model loads).
ENV PORT=8080 \
    TYPESTRICT_MODEL_DIR=/model \
    AWS_LWA_READINESS_CHECK_PATH=/healthz \
    AWS_LWA_READINESS_CHECK_HEALTHY_STATUS=200
LABEL org.opencontainers.image.source="https://github.com/typestrict-lab/serve" \
      org.opencontainers.image.title="typestrict-serve" \
      org.opencontainers.image.description="HTTP runtime for typestrict.ai models: Jev's decision API (POST /v1/systemone) on CPU. Add the model at /model." \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="typestrict.ai"
USER 10001:10001
WORKDIR /
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/typestrict-serve"]
