"""Typed failures with fixed, public-safe messages."""


class ProtocolError(Exception):
    status_code = 422
    code = "protocol_validation_error"
    public_message = "Invalid Protocol 2.0 request."


class ProtocolValidationError(ProtocolError):
    pass


class ImageNotFoundError(ProtocolError):
    status_code = 404
    code = "image_not_found"
    public_message = "Requested asset is unavailable."


class ImageHashMismatchError(ProtocolError):
    code = "image_hash_mismatch"
    public_message = "Asset metadata or integrity check failed."


class UnsupportedConstraintError(ProtocolError):
    code = "unsupported_constraint"
    public_message = "Unsupported answer constraint."


class AnswerContractViolation(ProtocolError):
    status_code = 500
    code = "answer_contract_violation"
    public_message = "Inference did not produce a valid constrained answer."


class BBoxValidationError(AnswerContractViolation):
    code = "bbox_validation_error"


class RuntimeInferenceError(ProtocolError):
    status_code = 500
    code = "runtime_inference_error"
    public_message = "Inference failed."
