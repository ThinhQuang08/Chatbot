pipeline {

    agent { label 'chatbot-mlops-214' }

    environment {
        PYTHON              = "/home/thinh/Chatbot_tien/.venv/bin/python"
        RUNTIME_PROJECT_DIR = "/home/thinh/Chatbot_tien"
        WORKSPACE_DIR       = "${WORKSPACE}"
        MODEL_DIR           = "${WORKSPACE}/rasa_bot/models"

        // Venv riêng cho các tool CI để không làm bẩn venv chính của chatbot
        CI_TOOLS_VENV       = "/home/jenkins/.cache/chatbot-ci-tools"

        // Lib path từng cần cho môi trường Rasa
        RASA_LIB_DIR        = "/home/thinh/miniconda3/envs/rasa/lib"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 90, unit: 'MINUTES')
        skipDefaultCheckout(true)
        parallelsAlwaysFailFast()
    }

    stages {

        stage('0. Checkout Source') {
            steps {
                echo "📥 Checkout source code..."
                deleteDir()
                checkout scm

                sh '''#!/usr/bin/env bash
                    set -euo pipefail

                    echo "Current workspace:"
                    pwd

                    echo "Latest commit:"
                    git log -1 --oneline || true

                    echo "Workspace size:"
                    du -sh . || true
                '''
            }
        }

        stage('1. Bootstrap CI Tools') {
            steps {
                echo "🧰 Chuẩn bị tool kiểm tra code/security..."

                sh '''#!/usr/bin/env bash
                    set -euo pipefail

                    mkdir -p "$(dirname "${CI_TOOLS_VENV}")"

                    if [ ! -x "${CI_TOOLS_VENV}/bin/python" ]; then
                        echo "Creating CI tools venv at ${CI_TOOLS_VENV}..."
                        "${PYTHON}" -m venv "${CI_TOOLS_VENV}" || python3 -m venv "${CI_TOOLS_VENV}"
                    fi

                    "${CI_TOOLS_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel

                    # ruff     : validate Python syntax + lint lỗi nghiêm trọng
                    # bandit   : scan security issue trong Python source code
                    # pip-audit: scan CVE/vulnerability trong dependency
                    "${CI_TOOLS_VENV}/bin/python" -m pip install --upgrade ruff bandit pip-audit

                    echo "CI tools versions:"
                    "${CI_TOOLS_VENV}/bin/ruff" --version
                    "${CI_TOOLS_VENV}/bin/bandit" --version
                    "${CI_TOOLS_VENV}/bin/pip-audit" --version
                '''
            }
        }

        stage('2. Quality Gates - Parallel') {
            parallel {

                stage('2.1 Verify Runtime Environment') {
                    steps {
                        echo "✅ Kiểm tra Python/Rasa/MLflow/SentenceTransformers..."

                        sh '''#!/usr/bin/env bash
                            set -euo pipefail

                            export LD_LIBRARY_PATH="${RASA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

                            if [ ! -x "${PYTHON}" ]; then
                                echo "❌ Không tìm thấy Python runtime: ${PYTHON}"
                                exit 1
                            fi

                            "${PYTHON}" --version

                            "${PYTHON}" - <<'PY'
import sys
print("Python executable:", sys.executable)

import rasa
print("Rasa:", rasa.__version__)

import mlflow
print("MLflow:", mlflow.__version__)

import sentence_transformers
print("SentenceTransformers:", sentence_transformers.__version__)
PY

                            echo "Disk status:"
                            df -h

                            echo "Memory status:"
                            free -h || true
                        '''
                    }
                }

                stage('2.2 Validate Python Code') {
                    steps {
                        echo "🐍 Validate Python source code..."

                        sh '''#!/usr/bin/env bash
                            set -euo pipefail

                            cd "${WORKSPACE_DIR}"
                            mkdir -p quality-reports

                            echo "Checking Python syntax with compileall..."

                            TARGETS=()
                            [ -d "data" ] && TARGETS+=("data")
                            [ -d "scripts" ] && TARGETS+=("scripts")
                            [ -d "rasa_bot" ] && TARGETS+=("rasa_bot")

                            if [ "${#TARGETS[@]}" -eq 0 ]; then
                                echo "⚠️ Không tìm thấy thư mục Python target: data/scripts/rasa_bot"
                                exit 1
                            fi

                            "${PYTHON}" -m compileall -q "${TARGETS[@]}"

                            echo "Running ruff critical lint..."
                            # E9  : syntax error
                            # F63 : invalid print/raise/assert style issues
                            # F7  : logic/control-flow issues
                            # F82 : undefined name
                            "${CI_TOOLS_VENV}/bin/ruff" check "${TARGETS[@]}" \
                                --select E9,F63,F7,F82 \
                                --output-format=github \
                                | tee quality-reports/ruff-critical.txt

                            echo "✅ Python validation passed."
                        '''
                    }
                }

                stage('2.3 Scan Dependencies & Security') {
                    steps {
                        echo "🛡️ Scan dependency, vulnerability và security issue..."

                        sh '''#!/usr/bin/env bash
                            set -euo pipefail

                            cd "${WORKSPACE_DIR}"
                            mkdir -p security-reports

                            echo "1) Export installed packages from runtime venv..."
                            "${PYTHON}" -m pip list --format=freeze | tee security-reports/installed-freeze.txt

                            echo "2) Validate dependency compatibility with pip check..."
                            "${PYTHON}" -m pip check | tee security-reports/pip-check.txt

                            echo "3) Audit known vulnerabilities with pip-audit..."

                            if [ -f "requirements.txt" ]; then
                                echo "Using requirements.txt for pip-audit..."
                                "${CI_TOOLS_VENV}/bin/pip-audit" \
                                    -r requirements.txt \
                                    --strict \
                                    -f json \
                                    -o security-reports/pip-audit.json
                            else
                                echo "requirements.txt not found. Auditing installed-freeze.txt instead..."
                                "${CI_TOOLS_VENV}/bin/pip-audit" \
                                    -r security-reports/installed-freeze.txt \
                                    --strict \
                                    -f json \
                                    -o security-reports/pip-audit.json
                            fi

                            echo "4) Static security scan Python code with Bandit..."

                            BANDIT_TARGETS=()
                            [ -d "data" ] && BANDIT_TARGETS+=("data")
                            [ -d "scripts" ] && BANDIT_TARGETS+=("scripts")
                            [ -d "rasa_bot" ] && BANDIT_TARGETS+=("rasa_bot")

                            if [ "${#BANDIT_TARGETS[@]}" -gt 0 ]; then
                                "${CI_TOOLS_VENV}/bin/bandit" \
                                    -r "${BANDIT_TARGETS[@]}" \
                                    -x "**/.venv/**,**/__pycache__/**,**/tests/**" \
                                    -ll -ii \
                                    -f json \
                                    -o security-reports/bandit.json
                            else
                                echo "⚠️ Không có target để Bandit scan."
                            fi

                            echo "5) Optional malware scan with ClamAV if available..."

                            if command -v clamscan >/dev/null 2>&1; then
                                clamscan -r \
                                    --infected \
                                    --exclude-dir="\\.git" \
                                    --exclude-dir="\\.venv" \
                                    --exclude-dir="__pycache__" \
                                    . | tee security-reports/clamscan.txt
                            else
                                echo "⚠️ clamscan chưa được cài trên agent. Bỏ qua malware file scan." | tee security-reports/clamscan.txt
                            fi

                            echo "✅ Dependency/security scan passed."
                        '''
                    }
                }
            }
        }

        stage('3. Prepare Runtime Files') {
            steps {
                echo "⚙️ Chuẩn bị runtime files..."

                sh '''#!/usr/bin/env bash
                    set -euo pipefail

                    cd "${WORKSPACE_DIR}"

                    if [ -f "${RUNTIME_PROJECT_DIR}/.env" ]; then
                        cp "${RUNTIME_PROJECT_DIR}/.env" .env
                        echo "✅ Copied .env from ${RUNTIME_PROJECT_DIR}/.env"
                    else
                        echo "❌ Không tìm thấy ${RUNTIME_PROJECT_DIR}/.env"
                        exit 1
                    fi

                    if grep -q "192.168.1.213" .env; then
                        echo "⚠️ Cảnh báo: .env vẫn có IP cũ 192.168.1.213. Hãy kiểm tra lại DB_HOST/MLFLOW/QDRANT nếu pipeline lỗi kết nối."
                    fi

                    mkdir -p "${MODEL_DIR}"

                    echo "Runtime files:"
                    ls -lah .env
                    ls -lah rasa_bot || true
                '''
            }
        }

        stage('4. Data Pipeline') {
            steps {
                echo "🧹 Chạy data pipeline..."

                sh '''#!/usr/bin/env bash
                    set -euo pipefail

                    cd "${WORKSPACE_DIR}"
                    export LD_LIBRARY_PATH="${RASA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

                    "${PYTHON}" data/generate_massive_data.py
                    "${PYTHON}" data/preprocess_data.py
                    "${PYTHON}" data/auto_label_snorkel.py
                    "${PYTHON}" data/split_confidence.py
                    "${PYTHON}" data/validate_cleanlab.py
                    "${PYTHON}" data/csv_to_rasa.py

                    echo "✅ Data pipeline completed."
                '''
            }
        }

        stage('5. Train Model') {
            steps {
                echo "🚀 Train Rasa model + log MLflow..."

                sh '''#!/usr/bin/env bash
                    set -euo pipefail

                    cd "${WORKSPACE_DIR}"
                    export LD_LIBRARY_PATH="${RASA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

                    "${PYTHON}" scripts/train_mlflow.py

                    echo "✅ Training completed."
                '''
            }
        }

        stage('6. Check Model Artifact') {
            steps {
                echo "🔎 Kiểm tra model artifact..."

                sh '''#!/usr/bin/env bash
                    set -euo pipefail

                    cd "${WORKSPACE_DIR}"

                    echo "Model directory:"
                    ls -lah "${MODEL_DIR}" || true

                    LATEST_MODEL="$(ls -t "${MODEL_DIR}"/*.tar.gz 2>/dev/null | head -n 1 || true)"

                    if [ -z "${LATEST_MODEL}" ]; then
                        echo "❌ Không tìm thấy model .tar.gz trong ${MODEL_DIR}"
                        exit 1
                    fi

                    echo "✅ Latest model: ${LATEST_MODEL}"
                    echo "${LATEST_MODEL}" > latest_model_path.txt
                '''
            }
        }

        stage('7. Human Approval') {
            steps {
                script {
                    def latestModel = sh(
                        script: 'cat latest_model_path.txt',
                        returnStdout: true
                    ).trim()

                    echo "🔔 Model đã train xong: ${latestModel}"
                    echo "📊 Vào MLflow kiểm tra metrics/report trước khi quyết định deploy."

                    def decision = input(
                        id: 'DeployGate',
                        message: "Deploy model này không?\n${latestModel}",
                        ok: 'Submit',
                        parameters: [
                            choice(
                                name: 'DECISION',
                                choices: ['deploy', 'reject'],
                                description: 'deploy = triển khai model, reject = dừng pipeline'
                            )
                        ]
                    )

                    if (decision == 'reject') {
                        currentBuild.result = 'ABORTED'
                        error("🛑 Model bị reject. Pipeline dừng, không deploy.")
                    }

                    echo "✅ Model được duyệt. Tiếp tục deploy."
                }
            }
        }

        stage('8. Deploy Model') {
            steps {
                echo "☁️ Deploy model..."

                sh '''#!/usr/bin/env bash
                    set -euo pipefail

                    cd "${WORKSPACE_DIR}"
                    export LD_LIBRARY_PATH="${RASA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

                    "${PYTHON}" scripts/deploy_model.py

                    echo "✅ Deploy completed."
                '''
            }
        }
    }

    post {
        always {
            echo "📦 Archive reports..."
            archiveArtifacts artifacts: '''
                latest_model_path.txt,
                quality-reports/**/*,
                security-reports/**/*
            ''', allowEmptyArchive: true
        }

        success {
            echo "🎉 PIPELINE HOÀN TẤT THÀNH CÔNG!"
        }

        aborted {
            echo "⚠️ Pipeline bị dừng. Có thể do model bị reject ở bước Human Approval."
        }

        failure {
            echo "🔥 Pipeline thất bại. Kiểm tra Console Output và các report đã archive."
        }
    }
}