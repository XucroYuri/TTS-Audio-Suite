from pathlib import Path
import importlib.util
import pytest
from types import SimpleNamespace
from api_bridge.models import TTSResource

spec = importlib.util.spec_from_file_location('checkpoint_bridge', Path(__file__).resolve().parents[2] / 'nodes/api_bridge/resource_engine_nodes.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


@pytest.mark.unit
def test_checkpoint_selection_is_confined_to_registered_directory(tmp_path):
    root = tmp_path / 'GPT_weights_v2'
    root.mkdir()
    registered = root / 'voice-e10.ckpt'
    selected = root / 'voice-e20.ckpt'
    registered.write_bytes(b'registered')
    selected.write_bytes(b'selected')
    assert bridge._checkpoint_choice('', registered, '.ckpt') == registered
    assert bridge._checkpoint_choice(selected.name, registered, '.ckpt') == selected
    other = tmp_path / 'outside.ckpt'
    other.write_bytes(b'outside')
    with pytest.raises(ValueError, match='registered weight directory'):
        bridge._checkpoint_choice(str(other), registered, '.ckpt')
    wrong_type = root / 'sovits.pth'
    wrong_type.write_bytes(b'wrong type')
    with pytest.raises(ValueError, match='registered weight directory'):
        bridge._checkpoint_choice(wrong_type.name, registered, '.ckpt')


@pytest.mark.unit
def test_both_selected_checkpoints_reach_the_adapter_configuration(tmp_path, monkeypatch):
    gpt_root = tmp_path/'GPT_weights'
    sovits_root = tmp_path/'SoVITS_weights'
    gpt_root.mkdir()
    sovits_root.mkdir()
    for root, names in ((gpt_root,('old.ckpt','selected.ckpt')),(sovits_root,('old.pth','selected.pth'))):
        for name in names:
            (root/name).write_bytes(b'checkpoint')
    resource = TTSResource(resource_id='registered',engine='gpt_sovits',source_root=tmp_path,gpt_weight=gpt_root/'old.ckpt',sovits_weight=sovits_root/'old.pth')
    monkeypatch.setattr(bridge,'get_resource_registry',lambda:SimpleNamespace(require=lambda *_:resource))
    (engine,) = bridge.ExternalGPTSovitsEngineNode().create_engine('registered',gpt_checkpoint='selected.ckpt',sovits_checkpoint='selected.pth')
    assert engine['config']['gpt_weight'] == str(gpt_root/'selected.ckpt')
    assert engine['config']['sovits_weight'] == str(sovits_root/'selected.pth')
    assert resource.gpt_weight.name == 'old.ckpt'
