"""
模块化注册表与热插拔管理器 (ModuleRegistry)

支持感知模块（从视觉开始，可扩展听觉、触觉、机械臂本体感知等）的动态注册、热插拔切换与版本升级。
"""

from __future__ import annotations

from typing import Dict, List, Optional
from .modules.base import PerceptionModule


class ModuleRegistry:
    """管理所有可插拔感官模块，支持按模态槽位热升级。"""

    def __init__(self) -> None:
        self._active_modules: Dict[str, PerceptionModule] = {}
        self._version_history: Dict[str, List[str]] = {}

    def register(self, module: PerceptionModule, replace: bool = True) -> None:
        slot = module.modality
        if slot in self._active_modules and not replace:
            raise ValueError(f"模态槽位 '{slot}' 已存在模块 {self._active_modules[slot].module_id}")
        self._active_modules[slot] = module
        self._version_history.setdefault(slot, []).append(
            f"{module.module_id}@{module.version}"
        )

    def upgrade(self, slot: str, new_module: PerceptionModule) -> str:
        """热升级指定模态槽位的模块，无需改动下游核心推理引擎。"""
        if new_module.modality != slot:
            raise ValueError(
                f"模块模态 '{new_module.modality}' 与目标槽位 '{slot}' 不匹配"
            )
        old_desc = (
            f"{self._active_modules[slot].module_id}@{self._active_modules[slot].version}"
            if slot in self._active_modules
            else "None"
        )
        self.register(new_module, replace=True)
        new_desc = f"{new_module.module_id}@{new_module.version}"
        return f"槽位 [{slot}] 模块热升级完成: {old_desc} -> {new_desc}"

    def get(self, slot: str) -> Optional[PerceptionModule]:
        return self._active_modules.get(slot)

    def active_modules(self) -> Dict[str, PerceptionModule]:
        return dict(self._active_modules)

    def status_summary(self) -> Dict[str, Dict[str, object]]:
        return {
            slot: {
                "active": f"{mod.module_id}@{mod.version}",
                "embedding_dim": mod.embedding_dim,
                "history": list(self._version_history.get(slot, [])),
            }
            for slot, mod in self._active_modules.items()
        }
