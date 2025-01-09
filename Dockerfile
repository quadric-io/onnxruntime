# syntax=docker/dockerfile:1.2

ARG TVM_IMAGE_TAG=stable
FROM ghcr.io/quadric-io/tvm:$TVM_IMAGE_TAG as tvm_base
