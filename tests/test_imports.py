"""
tests/test_imports.py
=====================
基础导入测试（Smoke Tests）。

确保核心示例模块可以无错误导入，捕获语法错误和
循环依赖等基础问题。

运行：python -m pytest tests/test_imports.py -v
"""

import sys
import importlib
import unittest
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCoreImports(unittest.TestCase):
    """测试核心模块可导入性。"""

    def test_import_numpy_scipy(self):
        """基础科学计算库。"""
        import numpy as np
        import scipy
        self.assertTrue(hasattr(np, 'array'))
        self.assertTrue(hasattr(scipy, '__version__'))

    def test_import_matplotlib(self):
        """绘图库。"""
        import matplotlib
        matplotlib.use('Agg')  # 无 GUI 后端
        import matplotlib.pyplot as plt
        self.assertTrue(callable(plt.figure))


class TestExampleImports(unittest.TestCase):
    """测试示例模块可导入性（不执行主逻辑）。"""

    def test_freshman_zero_to_one(self):
        """大一新生入门示例。"""
        spec = importlib.util.spec_from_file_location(
            "freshman_zero_to_one",
            PROJECT_ROOT / "examples" / "freshman_zero_to_one.py"
        )
        module = importlib.util.module_from_spec(spec)
        # 只加载不执行 __main__
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'HumanHand21'))
        self.assertTrue(hasattr(module, 'DexMVRetargeter'))

    def test_evaluation_framework(self):
        """评估框架。"""
        spec = importlib.util.spec_from_file_location(
            "evaluation_framework",
            PROJECT_ROOT / "examples" / "evaluation_framework.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'EvaluationMetrics'))
        self.assertTrue(hasattr(module, 'benchmark_all_methods'))

    def test_fk_ik_demo(self):
        """FK/IK 演示。"""
        spec = importlib.util.spec_from_file_location(
            "fk_ik_demo",
            PROJECT_ROOT / "examples" / "fk_ik_demo.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'PlanarArm2D'))

    def test_minimal_vla(self):
        """最小 VLA（需 torch，无环境时跳过）。"""
        try:
            import torch
        except ImportError:
            self.skipTest("torch 未安装，跳过 VLA 导入测试")
        spec = importlib.util.spec_from_file_location(
            "minimal_vla",
            PROJECT_ROOT / "examples" / "minimal_vla.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'MinimalVLA'))

    def test_rl_demo(self):
        """RL 演示。"""
        spec = importlib.util.spec_from_file_location(
            "rl_demo",
            PROJECT_ROOT / "examples" / "rl_demo.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'run_demo'))
        self.assertTrue(hasattr(module, 'run_train'))

    def test_world_model_demo(self):
        """世界模型演示。"""
        spec = importlib.util.spec_from_file_location(
            "world_model_demo",
            PROJECT_ROOT / "examples" / "world_model_demo.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'run_concept_demo'))


class TestDocsExistence(unittest.TestCase):
    """测试关键文档存在性。"""

    REQUIRED_DOCS = [
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "LICENSE",
        "requirements.txt",
        "setup/environment.yml",
        "docs/00-joint-concepts.md",
        "docs/01-what-is-vla.md",
        "docs/12-freshman-zero-to-one.md",
        "docs/13-vla-zero-to-one.md",
        "docs/14-rl-zero-to-one.md",
        "docs/15-world-model-zero-to-one.md",
        "docs/17-research-trends-and-positioning.md",
        "docs/18-frontier-papers-online.md",
    ]

    def test_required_docs_exist(self):
        """所有关键文档必须存在。"""
        for doc in self.REQUIRED_DOCS:
            path = PROJECT_ROOT / doc
            self.assertTrue(
                path.exists(),
                f"Required document missing: {doc}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
