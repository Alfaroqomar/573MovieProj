# MovieMatch commands. Run from the repository root.
# Configuration is read from .env (copy .env.example first).

PYTHON ?= python3
BACKEND = cd backend &&

.PHONY: install install-backend install-frontend status preprocess train neumf evaluate candidates backend frontend test build

install: install-backend install-frontend

install-backend:
	$(BACKEND) $(PYTHON) -m pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

status:
	$(BACKEND) $(PYTHON) -m moviematch.cli status

preprocess:
	$(BACKEND) $(PYTHON) -m moviematch.cli preprocess

train:
	$(BACKEND) $(PYTHON) -m moviematch.cli train

neumf:
	$(BACKEND) $(PYTHON) -m moviematch.cli train --only neumf

evaluate:
	$(BACKEND) $(PYTHON) -m moviematch.cli evaluate

candidates:
	$(BACKEND) $(PYTHON) -m moviematch.cli candidates

backend:
	$(BACKEND) $(PYTHON) -m uvicorn moviematch.api.main:app --host 127.0.0.1 --port 8000

frontend:
	cd frontend && npm run dev

test:
	$(BACKEND) $(PYTHON) -m pytest

build:
	cd frontend && npm run build
