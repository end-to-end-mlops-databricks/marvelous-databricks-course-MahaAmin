start:
	uv venv -p 3.11.11 .venv
	. .venv/bin/activate
	uv pip install -r pyproject.toml --all-extras
	uv lock

install:
	uv pip install -r pyproject.toml --all-extras

lock:
	uv pip install -r pyproject.toml --all-extras
	uv lock

pre-commit:
	uv run pre-commit

package:
	uv pip install -r pyproject.toml --all-extras
	uv lock
	uv build

db-push-pkg:
	databricks fs cp --overwrite dist/fraud_credit_cards-$(version)-py3-none-any.whl dbfs:/Volumes/fraud_credit_cards/packages/fraud_credit_cards
