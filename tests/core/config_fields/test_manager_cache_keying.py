"""
Tests for per-context caching of the config-field manager/generator singletons.

Regression for the Theme-4 hidden-mutable-state finding: these factories cached a single
global on first call, so a later call with DIFFERENT workspace_dirs / project_id silently
returned the stale (wrong-context) instance. They now cache per key.
"""

from cursus.core.config_fields.unified_config_manager import (
    get_unified_config_manager,
    reset_unified_config_manager_cache,
)
from cursus.core.config_fields.inheritance_aware_field_generator import (
    get_inheritance_aware_field_generator,
    reset_inheritance_aware_field_generator_cache,
)


class TestUnifiedConfigManagerCache:
    def setup_method(self):
        reset_unified_config_manager_cache()

    def teardown_method(self):
        reset_unified_config_manager_cache()

    def test_same_key_returns_same_instance(self):
        a = get_unified_config_manager(["/ws/a"])
        b = get_unified_config_manager(["/ws/a"])
        assert a is b

    def test_different_workspace_dirs_return_different_instances(self):
        a = get_unified_config_manager(["/ws/a"])
        b = get_unified_config_manager(["/ws/b"])
        assert a is not b
        assert a.workspace_dirs == ["/ws/a"]
        assert b.workspace_dirs == ["/ws/b"]

    def test_none_and_empty_share_a_key(self):
        assert get_unified_config_manager(None) is get_unified_config_manager([])

    def test_reset_clears_cache(self):
        a = get_unified_config_manager(["/ws/a"])
        reset_unified_config_manager_cache()
        assert get_unified_config_manager(["/ws/a"]) is not a


class TestFieldGeneratorCache:
    def setup_method(self):
        reset_inheritance_aware_field_generator_cache()

    def teardown_method(self):
        reset_inheritance_aware_field_generator_cache()

    def test_same_key_returns_same_instance(self):
        a = get_inheritance_aware_field_generator(["/ws/a"], "proj1")
        b = get_inheritance_aware_field_generator(["/ws/a"], "proj1")
        assert a is b

    def test_different_project_id_returns_different_instance(self):
        a = get_inheritance_aware_field_generator(["/ws/a"], "proj1")
        b = get_inheritance_aware_field_generator(["/ws/a"], "proj2")
        assert a is not b

    def test_different_workspace_dirs_return_different_instances(self):
        a = get_inheritance_aware_field_generator(["/ws/a"], "proj1")
        b = get_inheritance_aware_field_generator(["/ws/b"], "proj1")
        assert a is not b

    def test_reset_clears_cache(self):
        a = get_inheritance_aware_field_generator(["/ws/a"], "proj1")
        reset_inheritance_aware_field_generator_cache()
        assert get_inheritance_aware_field_generator(["/ws/a"], "proj1") is not a
