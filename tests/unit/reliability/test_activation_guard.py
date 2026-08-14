import pytest

from sat_rs_vlm.models.reliability.activation_guard import ActivationGuard


def test_activation_guard_records_non_finite_output() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.critical = torch.nn.Identity()

        def forward(self, value):
            return self.critical(value)

    model = TinyModel()
    guard = ActivationGuard(model, module_patterns=["critical"], max_abs=10.0)
    assert guard.install() == 1
    model(torch.tensor([float("nan")]))
    with pytest.raises(RuntimeError, match="blocked inference"):
        guard.assert_healthy()
    assert guard.report()["anomalies"][0]["reason"] == "non_finite"
    guard.close()


def test_activation_guard_accepts_normal_activation_and_detects_large_value() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Identity())
    guard = ActivationGuard(model, module_patterns=["0"], max_abs=2.0)
    guard.install()
    model(torch.tensor([1.0]))
    guard.assert_healthy()
    model(torch.tensor([3.0]))
    with pytest.raises(RuntimeError, match="max_abs_exceeded"):
        guard.assert_healthy()
    guard.close()


def test_activation_guard_research_mode_records_without_failing_condition() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Identity())
    guard = ActivationGuard(
        model,
        module_patterns=["0"],
        max_abs=2.0,
        mode="research",
    )
    guard.install()
    model(torch.tensor([float("inf")]))
    guard.assert_healthy()
    report = guard.report()
    assert report["mode"] == "research"
    assert report["guard_triggered"] is True
    guard.close()
