#!/bin/bash
set -eo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# default
DEFAULT_VENV_DIR="$PROJECT_ROOT/.venv"
DEFAULT_PYTHON_VERSION="3.10"

# user-defined
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
PYTHON_VERSION="${PYTHON_VERSION:-$DEFAULT_PYTHON_VERSION}"
declare -A INSTALLED_ENVS=()

create_env() {
    local env_name=$1
    local python_version=$2

    export UV_HTTP_TIMEOUT=1200

    echo -e "\n🌟 Creating environment for \033[1;34m${env_name}\033[0m..."

    local env_path="$VENV_DIR/$env_name"
    if [[ -d "$env_path" ]]; then
        echo -e "\n⚠️  Environment \033[1;31m${env_name}\033[0m already exists, skipping creation"
        INSTALLED_ENVS["$env_name"]=1
        source "$env_path/bin/activate"
        uv python pin $python_version
        return
    fi

    echo -e "🐍 Creating Python $python_version virtual environment..."
    if ! uv venv "$env_path" --python $python_version; then
        echo -e "❌ \033[1;31mFailed to create virtual environment\033[0m" >&2
        exit 1
    fi

    local python_path="$env_path/bin/python"
    if [[ ! -f "$python_path" ]]; then
        echo -e "❌ \033[1;31mPython executable not found in virtual environment\033[0m" >&2
        exit 1
    fi

    INSTALLED_ENVS["$env_name"]=1

    echo "Python version: $("$python_path" --version)"
    echo "Python path: $python_path"

    source "$env_path/bin/activate"
    uv python pin $python_version
}

##############################################################
# Install base requirements
##############################################################
install_base_requirements() {
    echo -e "\n📦 Installing framework base dependencies..."

    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo -e "❌ \033[1;31mNo virtual environment activated\033[0m" >&2
        exit 1
    fi

    cd "$PROJECT_ROOT"
    uv pip install -e . || {
        echo -e "❌ \033[1;31mFailed to install InternManip base dependencies\033[0m" >&2
        exit 1
    }
}

##############################################################
# Validate virtualenv directory
##############################################################
check_venv_dir() {
    if [[ ! -d "$VENV_DIR" ]]; then
        mkdir -p "$VENV_DIR"
        echo "📁 Created virtualenv directory: ${VENV_DIR}"
    fi
}

##############################################################
# Install Calvin benchmark dependencies
##############################################################
install_calvin() {
    check_venv_dir
    create_env "${CALVIN_ENV_NAME:-"calvin"}" "${PYTHON_VERSION}"

    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo -e "❌ \033[1;31mFailed to activate virtual environment\033[0m" >&2
        exit 1
    fi

    echo -e "\n🔗 Installing Calvin benchmark dependencies..."
    echo -e "📍 Virtual environment: ${VIRTUAL_ENV}"
    echo -e "📍 Python version: $(python --version)"

    local calvin_root="${PROJECT_ROOT}/internmanip/benchmarks/calvin"
    if [[ ! -d "$calvin_root" ]]; then
        echo -e "❌ \033[1;31mCalvin directory not found: ${calvin_root}\033[0m" >&2
        exit 1
    fi

    export CALVIN_ROOT="$calvin_root"
    cd "$CALVIN_ROOT" || { echo -e "❌ \033[1;31mFailed to enter Calvin benchmark root directory\033[0m" >&2; exit 1; }

    uv pip install wheel cmake==3.18.4
    cd calvin_env/tacto
    uv pip install -e .
    cd ..
    uv pip install -e .
    cd ../calvin_models
    uv pip install setuptools==57.5.0
    uv pip install "pip<23.0"
    pip install pyhash==0.9.3
    uv pip install -e .

    install_base_requirements

    uv pip install "networkx>=3.0"
    uv pip install numpy==1.23.5

    cd "$PROJECT_ROOT"

    deactivate
    echo -e "✅ \033[1;32mCalvin dependencies installed\033[0m"
}

##############################################################
# Install SimplerEnv benchmark dependencies
##############################################################
install_simpler_env() {
    check_venv_dir
    create_env "${SIMPLER_ENV_NAME:-"simpler_env"}" "${PYTHON_VERSION}"

    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo -e "❌ \033[1;31mFailed to activate virtual environment\033[0m" >&2
        exit 1
    fi

    echo -e "\n🔗 Installing SimplerEnv dependencies..."
    echo -e "📍 Virtual environment: ${VIRTUAL_ENV}"
    echo -e "📍 Python version: $(python --version)"

    local simpler_env_dir="${PROJECT_ROOT}/internmanip/benchmarks/SimplerEnv"
    if [[ ! -d "$simpler_env_dir" ]]; then
        echo -e "❌ \033[1;31mSimplerEnv directory not found: ${simpler_env_dir}\033[0m" >&2
        exit 1
    fi

    cd "${simpler_env_dir}/ManiSkill2_real2sim" || { echo -e "❌ \033[1;31mFailed to enter ManiSkill2_real2sim directory\033[0m" >&2; exit 1; }
    set -x
    uv pip install "setuptools==65.5.0"
    uv pip install -e . || { echo -e "❌ \033[1;31mFailed to install ManiSkill2_real2sim\033[0m" >&2; exit 1; }
    cd ..
    uv pip install -e . || { echo -e "❌ \033[1;31mFailed to install SimplerEnv\033[0m" >&2; exit 1; }
    uv pip install --no-cache-dir -r requirements_full_install.txt || { echo -e "⚠️  \033[1;33mWarning: Some SimplerEnv dependencies failed to install due to network issues\033[0m" >&2; }
    uv pip install torch transforms3d
    set +x

    install_base_requirements

    cd "$PROJECT_ROOT"

    deactivate
    echo -e "✅ \033[1;32mSimplerEnv dependencies installed\033[0m"
}

##############################################################
# Install GenManip benchmark dependencies
##############################################################
install_genmanip() {
    echo -e "\n🔗 Installing GenManip dependencies..."

    if [[ ! -d "internmanip/benchmarks/genmanip/utils/InternUtopia" ]]; then
        git submodule add --force https://github.com/InternRobotics/InternUtopia.git internmanip/benchmarks/genmanip/utils/InternUtopia
    fi

    cd $PROJECT_ROOT/internmanip/benchmarks/genmanip/utils/InternUtopia
    git checkout tags/v2.2.1

    echo "If you haven't installed Isaac Sim yet, please do so before running this setup script."
    read -p "If you have already installed it in a custom location, please type in the path containing isaac-sim.sh here >>> " ISAAC_SIM_PATH

    {
        echo "$ISAAC_SIM_PATH"
        echo "genmanip"
    } | ./setup_conda.sh

    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate genmanip

    cd $PROJECT_ROOT/internmanip/benchmarks/genmanip/utils/curobo
    pip install -e . --no-build-isolation  
    
    cd $PROJECT_ROOT 
    pip install -e .

    pip install lmdb==1.6.2 open3d==0.19.0 shapely==2.1.1 concave_hull==0.0.9 roboticstoolbox-python==1.1.1

    conda deactivate
    echo -e "Use \033[1;33mconda activate genmanip\033[0m to activate the environment"
    echo -e "✅ \033[1;32mGenManip dependencies installed\033[0m"
}

##############################################################
# Install model dependencies
##############################################################
install_model() {
    check_venv_dir
    create_env "${MODEL_ENV_NAME:-"model"}" "${PYTHON_VERSION}"
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo -e "❌ \033[1;31mFailed to activate virtual environment\033[0m" >&2
        exit 1
    fi

    echo -e "🔍 Checking system requirements..."

    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            case "$VERSION_ID" in
                "20.04"|"22.04")
                    echo -e "✅ Ubuntu version check passed: $VERSION_ID"
                    ;;
                *)
                    echo -e "⚠️  \033[1;33mWarning: Current Ubuntu version is $VERSION_ID, recommended to use 20.04 or 22.04\033[0m"
                    ;;
            esac
        else
            echo -e "⚠️  \033[1;33mWarning: Current system is not Ubuntu, recommended to use Ubuntu 20.04 or 22.04\033[0m"
        fi
    else
        echo -e "⚠️  \033[1;33mWarning: Unable to detect system version\033[0m"
    fi

    cuda_detected=false
    cuda_version=""

    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        cuda_detected=true
        echo -e "📍 CUDA detected via nvcc: $cuda_version"
    fi

    if command -v nvidia-smi &> /dev/null; then
        runtime_cuda=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        if [[ -n "$runtime_cuda" ]]; then
            echo -e "📍 CUDA runtime detected via nvidia-smi: $runtime_cuda"
            if [[ "$cuda_detected" == false ]]; then
                cuda_version="$runtime_cuda"
                cuda_detected=true
            fi
        fi
    fi

    if [[ "$cuda_detected" == true ]]; then
        cuda_major_minor=$(echo "$cuda_version" | cut -d'.' -f1,2)
        if [[ "$cuda_major_minor" == "12.4" ]]; then
            echo -e "✅ CUDA version check passed: $cuda_version"
        else
            echo -e "⚠️  \033[1;33mWarning: Current CUDA version is $cuda_version, recommended to use CUDA 12.4\033[0m"
        fi
    else
        echo -e "⚠️  \033[1;33mWarning: CUDA not detected. Please ensure CUDA 12.4 is installed\033[0m"
    fi

    echo -e "\n🔍 Checking additional system dependencies required by Gr00t models: ffmpeg, libsm6, libxext6..."

    if command -v ffmpeg &> /dev/null; then
        echo -e "✅ ffmpeg is installed"
    else
        echo -e "⚠️  \033[1;33mWarning: ffmpeg is not installed. Please install it using: sudo apt-get install ffmpeg\033[0m"
    fi

    if dpkg -l | grep -q libsm6; then
        echo -e "✅ libsm6 is installed"
    else
        echo -e "⚠️  \033[1;33mWarning: libsm6 is not installed. Please install it using: sudo apt-get install libsm6\033[0m"
    fi

    if dpkg -l | grep -q libxext6; then
        echo -e "✅ libxext6 is installed"
    else
        echo -e "⚠️  \033[1;33mWarning: libxext6 is not installed. Please install it using: sudo apt-get install libxext6\033[0m"
    fi

    echo -e "\n📋 To install all missing dependencies at once, run:"
    echo -e "   sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6"

    echo -e "\n🔍 Installing model dependencies..."
    echo -e "📍 Virtual environment: ${VIRTUAL_ENV}"
    echo -e "📍 Python version: $(python --version)"
    uv pip install --upgrade setuptools
    uv pip install "git+https://github.com/NVIDIA/Isaac-GR00T.git#egg=isaac-gr00t[base]" || { echo -e "⚠️  \033[1;33mWarning: Failed to install GR00T dependencies due to network issues\033[0m" >&2; }
    echo -e "📦 Installing flash-attn module..."
    uv pip install --no-build-isolation flash-attn==2.7.1.post4 || { echo -e "⚠️  \033[1;33mWarning: Failed to install flash-attn due to network issues\033[0m" >&2; }
    uv pip install draccus
    uv pip install transforms3d

    install_base_requirements

    deactivate
    echo -e "✅ \033[1;32mModel dependencies installed\033[0m"
}

##############################################################
# Install model dependencies by building from source
##############################################################
install_model_bfs() {
    check_venv_dir
    create_env "${MODEL_BFS_ENV_NAME:-"model_bfs"}" "${PYTHON_VERSION}"
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo -e "❌ \033[1;31mFailed to activate virtual environment\033[0m" >&2
        exit 1
    fi

    echo -e "\n🔧 Installing Isaac-GR00T from source (build from source)..."
    TMP_DIR=$(mktemp -d)
    echo "Using temporary directory: $TMP_DIR"
    cd "$TMP_DIR" || { echo -e "❌ \033[1;31mFailed to enter temporary directory\033[0m" >&2; exit 1; }

    if ! git clone https://github.com/NVIDIA/Isaac-GR00T.git; then
        echo -e "❌ \033[1;31mFailed to clone Isaac-GR00T repository\033[0m" >&2
        cd "$PROJECT_ROOT"
        exit 1
    fi

    mv Isaac-GR00T .gr00t

    cd .gr00t || { echo -e "❌ \033[1;31mFailed to enter .gr00t directory\033[0m" >&2; cd "$PROJECT_ROOT"; exit 1; }

    uv pip install --upgrade setuptools || { echo -e "❌ \033[1;31mFailed to upgrade setuptools\033[0m" >&2; cd "$PROJECT_ROOT"; exit 1; }
    uv pip install -e .[base] || { echo -e "❌ \033[1;31mFailed to install Isaac-GR00T base dependencies\033[0m" >&2; cd "$PROJECT_ROOT"; exit 1; }
    uv pip install --no-build-isolation flash-attn==2.7.1.post4 || { echo -e "❌ \033[1;31mFailed to install flash-attn module\033[0m" >&2; cd "$PROJECT_ROOT"; exit 1; }

    cd "$PROJECT_ROOT" || exit 1

    uv pip install draccus
    uv pip install transforms3d

    install_base_requirements

    deactivate
    echo -e "✅ \033[1;32mmodel (build from source) dependencies installed\033[0m"
}

##############################################################
# Main entry point
##############################################################
main() {
    local install_all=false
    local install_beginner=false

    if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]]; then
        echo -e "\033[1mUsage:\033[0m"
        echo "  --calvin [NAME]         Create Calvin benchmark virtual environment and install dependencies"
        echo "  --simpler-env [NAME]    Create SimplerEnv benchmark virtual environment and install dependencies"
        echo "  --genmanip [NAME]       Create GenManip benchmark virtual environment and install dependencies"
        echo "  --model [NAME]          Create model virtual environment and install dependencies"
        echo "  --model_bfs [NAME]      Create model virtual environment and build from source"
        echo "  --all                   Create all virtual environments and install dependencies (recommended for advanced users)"
        echo "  --beginner              Create beginner virtual environments and install dependencies (without genmanip, recommended for beginners)"
        echo ""
        echo -e "\033[1mCustomization Options:\033[0m"
        echo "  --venv-dir DIR          Set custom virtual environment root directory (default: .venv)"
        echo "  --python-version V      Set default Python version (recommended default: 3.10)"
        echo ""
        echo -e "\033[1mExamples:\033[0m"
        echo "  ./install.sh --venv-dir ./my_envs --model"
        echo "  ./install.sh --calvin calvin-test --model model-test"
        echo "  ./install.sh --python-version 3.10 --calvin calvin-dev --simpler-env simpler-dev"
        echo "  ./install.sh --all"
        echo "  ./install.sh --beginner"
        echo "  ./install.sh --model_bfs"
        echo "  --help                  Show help information"
        exit 0
    fi

    local args=("$@")
    local install_commands=()
    local i=0

    while (( i < ${#args[@]} )); do
        case "${args[i]}" in
            --venv-dir)
                VENV_DIR="${args[i+1]}"
                ((i += 2))
                ;;
            --python-version)
                PYTHON_VERSION="${args[i+1]}"
                ((i += 2))
                ;;
            --calvin|--simpler-env|--genmanip|--model|--model_bfs|--all|--beginner)
                install_commands+=("${args[i]}")
                if [[ -n "${args[i+1]}" && "${args[i+1]}" != --* ]]; then
                    install_commands+=("${args[i+1]}")
                    ((i += 2))
                else
                    ((i += 1))
                fi
                ;;
            *)
                echo -e "❌ \033[1;31mInvalid option: ${args[i]}\033[0m" >&2
                exit 1
                ;;
        esac
    done

    # Second pass: handle installation commands
    i=0
    while (( i < ${#install_commands[@]} )); do
        case "${install_commands[i]}" in
            --calvin)
                if [[ -n "${install_commands[i+1]}" && "${install_commands[i+1]}" != --* ]]; then
                    CALVIN_ENV_NAME="${install_commands[i+1]}"
                    ((i += 2))
                else
                    ((i += 1))
                fi
                install_calvin
                ;;
            --simpler-env)
                if [[ -n "${install_commands[i+1]}" && "${install_commands[i+1]}" != --* ]]; then
                    SIMPLER_ENV_NAME="${install_commands[i+1]}"
                    ((i += 2))
                else
                    ((i += 1))
                fi
                install_simpler_env
                ;;
            --genmanip)
                if [[ -n "${install_commands[i+1]}" && "${install_commands[i+1]}" != --* ]]; then
                    GENMANIP_ENV_NAME="${install_commands[i+1]}"
                    ((i += 2))
                else
                    ((i += 1))
                fi
                install_genmanip
                ;;
            --model)
                if [[ -n "${install_commands[i+1]}" && "${install_commands[i+1]}" != --* ]]; then
                    MODEL_ENV_NAME="${install_commands[i+1]}"
                    ((i += 2))
                else
                    ((i += 1))
                fi
                install_model
                ;;
            --model_bfs)
                if [[ -n "${install_commands[i+1]}" && "${install_commands[i+1]}" != --* ]]; then
                    MODEL_BFS_ENV_NAME="${install_commands[i+1]}"
                    ((i += 2))
                else
                    ((i += 1))
                fi
                install_model_bfs
                ;;
            --all)
                install_all=true
                ((i += 1))
                ;;
            --beginner)
                install_beginner=true
                ((i += 1))
                ;;
        esac
    done

    if [[ "$install_all" == true ]]; then
        install_calvin
        install_simpler_env
        install_genmanip
        install_model
    fi

    if [[ "$install_beginner" == true ]]; then
        install_calvin
        install_simpler_env
        install_model
    fi

    echo -e "\n🎉 \033[1;32mInstallation completed!\033[0m"
    echo -e "\033[1mConfiguration:\033[0m"
    echo -e "  📁 Virtual environment root directory: ${VENV_DIR}"
    echo -e "  🐍 Python version: ${PYTHON_VERSION}"
    echo -e "\033[1mNext steps:\033[0m"
    for env in "${!INSTALLED_ENVS[@]}"; do
        echo -e "  📌 Activate environment ${env}: source ${VENV_DIR}/${env}/bin/activate"
    done
}

main "$@"