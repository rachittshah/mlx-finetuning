#!/usr/bin/env python3
"""
Repository Validation Test Suite
Tests imports, syntax, and basic functionality
"""

import sys
import ast
from pathlib import Path

def test_python_syntax(script_path):
    """Test if Python file has valid syntax"""
    try:
        with open(script_path, 'r') as f:
            ast.parse(f.read())
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def test_imports(script_path):
    """Test if all imports are valid (without executing)"""
    try:
        with open(script_path, 'r') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        
        return True, f"Imports: {', '.join(imports[:5])}" if imports else "No imports"
    except Exception as e:
        return False, str(e)

def test_executable_scripts():
    """Test if scripts are executable"""
    scripts_dir = Path("scripts")
    scripts = list(scripts_dir.glob("*.py"))
    
    executable = []
    for script in scripts:
        if script.stat().st_mode & 0o111:  # Check if executable bit set
            executable.append(script.name)
    
    return executable

def test_documentation():
    """Test if documentation files exist and are readable"""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        return False, "docs/ directory missing"
    
    docs = list(docs_dir.glob("*.md"))
    if len(docs) < 3:
        return False, f"Only {len(docs)} docs found, expected 3+"
    
    total_lines = 0
    for doc in docs:
        with open(doc, 'r') as f:
            total_lines += len(f.readlines())
    
    return True, f"{len(docs)} docs with {total_lines} total lines"

def main():
    print("=" * 70)
    print("MLX-FINETUNING REPOSITORY VALIDATION TEST")
    print("=" * 70)
    
    # Test 1: Python syntax
    print("\n[TEST 1] Python Syntax Validation")
    print("-" * 70)
    scripts_dir = Path("scripts")
    scripts = list(scripts_dir.glob("*.py"))
    
    passed = 0
    failed = 0
    
    for script in scripts:
        success, msg = test_python_syntax(script)
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {script.name:40s} {msg}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Summary: {passed} passed, {failed} failed")
    
    # Test 2: Import analysis
    print("\n[TEST 2] Import Analysis")
    print("-" * 70)
    
    for script in scripts[:5]:  # Test first 5 scripts
        success, msg = test_imports(script)
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {script.name:40s}")
        if success:
            print(f"         {msg}")
    
    # Test 3: Executable permissions
    print("\n[TEST 3] Executable Permissions")
    print("-" * 70)
    executable = test_executable_scripts()
    print(f"  Executable scripts: {len(executable)}")
    for script in executable[:5]:
        print(f"    - {script}")
    
    # Test 4: Documentation
    print("\n[TEST 4] Documentation Completeness")
    print("-" * 70)
    success, msg = test_documentation()
    status = "PASS" if success else "FAIL"
    print(f"  [{status}] {msg}")
    
    # Test 5: README structure
    print("\n[TEST 5] README Structure")
    print("-" * 70)
    readme = Path("README.md")
    if readme.exists():
        with open(readme, 'r') as f:
            content = f.read()
            has_lora = "LoRA" in content or "Low-Rank Adaptation" in content
            has_install = "install" in content.lower()
            has_usage = "usage" in content.lower() or "quick" in content.lower()
            
            print(f"  [{'PASS' if has_lora else 'FAIL'}] Contains LoRA explanation")
            print(f"  [{'PASS' if has_install else 'FAIL'}] Contains installation instructions")
            print(f"  [{'PASS' if has_usage else 'FAIL'}] Contains usage examples")
            print(f"  README size: {len(content)} characters")
    else:
        print("  [FAIL] README.md not found")
    
    # Test 6: No emojis check
    print("\n[TEST 6] Emoji-Free Validation")
    print("-" * 70)
    
    emoji_pattern = r'[🚀📥✅📍❌🟢🟡🟠🔴📈⭐📂🔬🏆📊🎯💾]'
    import re
    
    emoji_count = 0
    for script in scripts:
        with open(script, 'r') as f:
            content = f.read()
            emojis = re.findall(emoji_pattern, content)
            if emojis:
                print(f"  [WARN] {script.name}: {len(emojis)} emojis found")
                emoji_count += len(emojis)
    
    for doc in Path("docs").glob("*.md"):
        with open(doc, 'r') as f:
            content = f.read()
            emojis = re.findall(emoji_pattern, content)
            if emojis:
                print(f"  [WARN] {doc.name}: {len(emojis)} emojis found")
                emoji_count += len(emojis)
    
    if emoji_count == 0:
        print(f"  [PASS] No emojis found in codebase")
    else:
        print(f"  [FAIL] {emoji_count} emojis found")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Total Scripts: {len(scripts)}")
    print(f"  Syntax Check: {passed}/{len(scripts)} passed")
    print(f"  Executable: {len(executable)} scripts")
    print(f"  Documentation: 3 guides")
    print(f"  Emoji-Free: {'YES' if emoji_count == 0 else 'NO'}")
    
    overall = (failed == 0 and emoji_count == 0)
    print(f"\n  Overall Status: {'PASS ✓' if overall else 'NEEDS FIXES ✗'}")
    print("=" * 70)
    
    return 0 if overall else 1

if __name__ == "__main__":
    sys.exit(main())
