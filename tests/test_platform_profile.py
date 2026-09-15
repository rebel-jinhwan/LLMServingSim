"""Every registered platform exposes a PlatformProfile, a step platform whose
profile has no grid is refused, and an unbound scheduler falls back to the
in-tree port. Was ``python -m platforms.profile``."""


def test_platform_profile() -> None:
    """Every registered platform's profile is a ``PlatformProfile``, and a
    step platform whose profile has no grid is refused."""
    from platforms import registry
    from platforms.profile import PlatformProfile as Base
    from platforms.spec import PlatformSpec

    for spec in registry().values():
        assert isinstance(spec.profile, Base), (spec.name, type(spec.profile))

    class NoGrid(Base):
        def measure(self, run_forward, iterations, catalog_slice):
            return []

    class NoGridPlatform(PlatformSpec):
        name, granularity = "nogrid", "step"
        profile_cls = NoGrid  # a plain class attribute satisfies the property's contract

    try:
        NoGridPlatform().profile
    except TypeError as e:
        assert "step_grid" in str(e), e
    else:
        raise AssertionError("a step platform without step_grid was accepted")
    print(f"ok: {sorted(registry())} expose PlatformProfile; a step platform without step_grid is refused")


def test_scheduler_binding() -> None:
    """A platform binds its scheduler by assigning self.scheduler inside
    bind_scheduler(); one that assigns nothing takes the in-tree port, and
    assigning something that is not a class is refused on the spot."""
    from platforms.spec import PlatformSpec
    from serving.core.scheduler import Scheduler

    class NoScheduler(PlatformSpec):
        name = "nosched"

    class OwnScheduler(PlatformSpec):
        name = "ownsched"

        def bind_scheduler(self):
            self.scheduler = Scheduler  # any class; the point is that it sticks

    class BadScheduler(PlatformSpec):
        name = "badsched"

        def bind_scheduler(self):
            self.scheduler = "serving.core.scheduler.Scheduler"

    assert NoScheduler().bind_scheduler() is None
    assert NoScheduler().scheduler is Scheduler
    own = OwnScheduler()
    assert own.scheduler is Scheduler and own.scheduler is own.scheduler
    try:
        BadScheduler().scheduler
    except TypeError as e:
        assert "must be a class" in str(e), e
    else:
        raise AssertionError("a non-class scheduler binding was accepted")
    print("ok: an unbound scheduler falls back to the in-tree port; a non-class binding is refused")
