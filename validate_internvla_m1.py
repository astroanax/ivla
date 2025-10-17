#!/usr/bin/env python3
"""
Validation script for InternVLA-M1 integration.

This script verifies that all components are properly installed and configured.

Usage:
    python validate_internvla_m1.py
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("=" * 60)
    print("Checking Imports...")
    print("=" * 60)
    
    checks = []
    
    # Check InternManip modules
    try:
        from internmanip.configs.model.internvla_m1_cfg import InternVLA_M1_Config
        print("✓ InternVLA_M1_Config imported")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Failed to import InternVLA_M1_Config: {e}")
        checks.append(False)
    
    try:
        from internmanip.model.basemodel.transforms.internvla_m1 import InternVLAM1Transform
        print("✓ InternVLAM1Transform imported")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Failed to import InternVLAM1Transform: {e}")
        checks.append(False)
    
    try:
        from internmanip.model.basemodel.internvla_m1 import InternVLA_M1
        print("✓ InternVLA_M1 model imported")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Failed to import InternVLA_M1: {e}")
        checks.append(False)
    
    try:
        from internmanip.agent.internvla_m1_agent import InternVLAM1Agent
        print("✓ InternVLAM1Agent imported")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Failed to import InternVLAM1Agent: {e}")
        checks.append(False)
    
    # Check InternVLA-M1 base modules
    try:
        from InternVLA.model.framework.M1 import InternVLA_M1 as InternVLA_M1_Base
        print("✓ InternVLA-M1 base model imported")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Failed to import InternVLA-M1 base: {e}")
        print("  → Ensure InternVLA-M1 is in parent directory or PYTHONPATH")
        checks.append(False)
    
    return all(checks)


def check_registry():
    """Check if InternVLA-M1 is registered in the agent registry."""
    print("\n" + "=" * 60)
    print("Checking Agent Registry...")
    print("=" * 60)
    
    try:
        from internmanip.agent.base import AgentRegistry
        
        # Check if INTERNVLA_M1 is in the registry
        if hasattr(AgentRegistry, 'INTERNVLA_M1'):
            print("✓ InternVLA-M1 found in AgentRegistry")
            
            # Try to access the agent class
            try:
                agent_class = AgentRegistry.INTERNVLA_M1.value
                print(f"✓ Agent class accessible: {agent_class.__name__}")
                return True
            except Exception as e:
                print(f"✗ Failed to access agent class: {e}")
                return False
        else:
            print("✗ InternVLA-M1 not found in AgentRegistry")
            print("  Available agents:", [a.name for a in AgentRegistry])
            return False
    except Exception as e:
        print(f"✗ Error checking registry: {e}")
        return False


def check_model_registration():
    """Check if InternVLA-M1 is registered with transformers."""
    print("\n" + "=" * 60)
    print("Checking Model Registration...")
    print("=" * 60)
    
    try:
        from transformers import AutoConfig, AutoModel
        
        # Try to get config
        try:
            config = AutoConfig.for_model('internvla_m1')
            print("✓ InternVLA-M1 config registered with AutoConfig")
        except Exception as e:
            print(f"✗ Config not registered: {e}")
            return False
        
        print("✓ Model registration successful")
        return True
    except Exception as e:
        print(f"✗ Error checking model registration: {e}")
        return False


def check_config_file():
    """Check if evaluation config file exists."""
    print("\n" + "=" * 60)
    print("Checking Configuration Files...")
    print("=" * 60)
    
    config_path = Path("challenge/run_configs/eval/internvla_m1_on_real_dummy.py")
    
    if config_path.exists():
        print(f"✓ Evaluation config found: {config_path}")
        
        # Try to load it
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("eval_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            if hasattr(config_module, 'eval_cfg'):
                print("✓ Config loads successfully")
                print(f"  - Agent type: {config_module.eval_cfg.agent.agent_type}")
                print(f"  - Eval type: {config_module.eval_cfg.agent.eval_type}")
                return True
            else:
                print("✗ Config missing 'eval_cfg' attribute")
                return False
        except Exception as e:
            print(f"✗ Failed to load config: {e}")
            return False
    else:
        print(f"✗ Config file not found: {config_path}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Checking Dependencies...")
    print("=" * 60)
    
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'PIL',
        'omegaconf',
        'qwen_vl_utils',
    ]
    
    checks = []
    for package in required_packages:
        try:
            __import__(package.replace('_', '-'))
            print(f"✓ {package} installed")
            checks.append(True)
        except ImportError:
            print(f"✗ {package} not installed")
            checks.append(False)
    
    return all(checks)


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("InternVLA-M1 Integration Validation")
    print("=" * 60)
    
    results = {
        'Imports': check_imports(),
        'Dependencies': check_dependencies(),
        'Agent Registry': check_registry(),
        'Model Registration': check_model_registration(),
        'Config Files': check_config_file(),
    }
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! InternVLA-M1 is ready to use.")
        print("\nNext steps:")
        print("1. Start agent server: python -m scripts.eval.start_agent_server --ports 5000")
        print("2. Run evaluation: python -m challenge.scripts.start_dummy_evaluator \\")
        print("                   --config challenge/run_configs/eval/internvla_m1_on_real_dummy.py \\")
        print("                   --server")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r InternVLA-M1/requirements.txt")
        print("- Add to PYTHONPATH: export PYTHONPATH=\"$PWD/InternVLA-M1:$PYTHONPATH\"")
        print("- Check file paths and directory structure")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()
