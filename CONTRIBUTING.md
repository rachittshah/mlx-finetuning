# Contributing to MLX Fine-Tuning

Thank you for considering contributing to MLX Fine-Tuning! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/mlx-finetuning.git
cd mlx-finetuning
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep lines under 100 characters when possible
- No emojis in code or documentation

## Testing

Before submitting a PR, run the validation test:
```bash
python3 test_repo.py
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run validation tests
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature`
7. Open a Pull Request

## Reporting Issues

When reporting issues, please include:
- Your environment (macOS version, Python version, MLX version)
- Steps to reproduce
- Expected vs actual behavior
- Error messages and logs

## Feature Requests

We welcome feature requests! Please open an issue describing:
- The use case
- How it would work
- Why it would be useful

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
