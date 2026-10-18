#!/usr/bin/env bash
# Ubuntu VM (no Docker): dev tools + Python/JAX/ipython + Julia/PythonCall.
# Mirrors the TrixiDocker :jax image. Usage: bash vm_setup_jax.sh        (CPU)
#                                           ACCEL=tpu bash vm_setup_jax.sh   (TPU VM, installs jax[tpu])
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
NVIM_VERSION=v0.12.5
JULIA_VERSION=1.12.7
PY_VERSION=3.14.8
PARSERS="bash c diff html lua luadoc markdown markdown_inline query vim vimdoc python yaml latex typst"
ME=$(id -un)
ACCEL=${ACCEL:-cpu}
# TPU VM images preload tcmalloc, which crashes Julia
unset LD_PRELOAD

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates curl git gcc g++ make libc6-dev unzip xz-utils zsh tmux fzf zoxide direnv \
  bat ripgrep fd-find file less ncurses-term kitty-terminfo universal-ctags global htop
sudo ln -sf /usr/bin/batcat /usr/local/bin/bat
sudo ln -sf /usr/bin/fdfind /usr/local/bin/fd
# Node 22 from NodeSource (Ubuntu 24.04's apt Node is 18, too old for claude-code etc.)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm i -g tree-sitter-cli pyright || echo 'WARN: npm global install (tree-sitter-cli, pyright) failed'

# rclone (config is NOT part of this script; copy ~/.config/rclone/rclone.conf yourself)
curl -fsSL https://rclone.org/install.sh | sudo bash

# neovim
curl -fsSL https://github.com/neovim/neovim/releases/download/${NVIM_VERSION}/nvim-linux-x86_64.tar.gz | sudo tar -xz -C /opt
sudo rm -rf /opt/nvim && sudo mv /opt/nvim-linux-x86_64 /opt/nvim
sudo ln -sf /opt/nvim/bin/nvim /usr/local/bin/nvim

# eza / atuin / yazi
T=$(mktemp -d); cd "$T"
base() { echo "https://github.com/$1/releases/latest/download/$2"; }
curl -fsSL "$(base eza-community/eza eza_x86_64-unknown-linux-gnu.tar.gz)" | sudo tar -xz -C /usr/local/bin
curl -fsSL "$(base atuinsh/atuin atuin-x86_64-unknown-linux-gnu.tar.gz)" | tar -xz -C "$T"
sudo cp "$(find "$T" -name atuin -type f | head -1)" /usr/local/bin/atuin
curl -fsSL -o y.zip "$(base sxyazi/yazi yazi-x86_64-unknown-linux-gnu.zip)"
unzip -j y.zip '*/yazi' '*/ya' -d "$T/y"
sudo install -m755 "$T"/y/yazi "$T"/y/ya /usr/local/bin/
cd ~

# configs
mkdir -p ~/.config
[ -d ~/.config/nvim ] || git clone --depth=1 https://github.com/junyixu/nvim-config.git ~/.config/nvim
[ -d ~/.config/zsh ]  || git clone --depth=1 https://github.com/junyixu/zsh-config-public.git ~/.config/zsh
[ -d ~/.config/yazi ] || git clone --depth=1 https://github.com/junyixu/yazi-config-public.git ~/.config/yazi
echo 'for f in ~/.config/zsh/zshrc.d/*.zsh(N); do source "$f"; done && typeset -U path PATH' > ~/.zshrc

nvim --headless "+Lazy! install" +qa
nvim --headless -c "lua require('nvim-treesitter').install(vim.split('${PARSERS}', ' ')):wait(900000)" -c qa
for i in 1 2 3; do
  nvim --headless \
    -c "lua require('lazy').load({ plugins = { 'blink.cmp' } })" \
    -c "lua local f = vim.fn.stdpath('data') .. '/lazy/blink.cmp/target/release/version'; print(vim.wait(120000, function() local h = io.open(f); if not h then return false end; local v = h:read('*a'); h:close(); return v ~= '' and v ~= 'v0.0.0' end, 200))" \
    -c "lua vim.wait(1000)" -c qa 2>&1 || true
  v=$(cat ~/.local/share/nvim/lazy/blink.cmp/target/release/version 2>/dev/null || true)
  [ -n "$v" ] && [ "$v" != v0.0.0 ] && break
done
TERM=xterm-256color zsh -ic 'exit' </dev/null || true
~/.local/share/zinit/plugins/romkatv---powerlevel10k/gitstatus/install || true
sudo chsh -s /usr/bin/zsh "$ME"

# uv + Python + JAX + ipython
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_UNMANAGED_INSTALL=/usr/local/bin sh
sudo mkdir -p /opt/uv-python /opt/jax-env && sudo chown "$ME": /opt/uv-python /opt/jax-env
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
uv venv /opt/jax-env --python ${PY_VERSION}
if [ "$ACCEL" = tpu ]; then JAXPKG='jax[tpu]==0.11.0'; else JAXPKG='jax==0.11.0'; fi
uv pip install --python /opt/jax-env/bin/python "$JAXPKG" ipython

# Julia + PythonCall (existing Python, no Conda)
curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | sudo tar -xz -C /opt
sudo rm -rf /opt/julia && sudo mv /opt/julia-${JULIA_VERSION} /opt/julia

cat <<'EOF' | sudo tee /etc/profile.d/jax.sh >/dev/null
export JULIA_PROJECT="$HOME/jax-julia-env"
export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)"
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/opt/jax-env/bin/python
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
case ":$PATH:" in *:/opt/julia/bin:*) ;; *) export PATH="/opt/jax-env/bin:/opt/julia/bin:$HOME/.julia/bin:$PATH" ;; esac
EOF
[ "$ACCEL" = tpu ] || echo 'export JAX_PLATFORMS=cpu' | sudo tee -a /etc/profile.d/jax.sh >/dev/null
echo '[ -r /etc/profile.d/jax.sh ] && emulate sh -c ". /etc/profile.d/jax.sh"' | sudo tee -a /etc/zsh/zshenv >/dev/null
. /etc/profile.d/jax.sh
mkdir -p "$JULIA_PROJECT"
julia -e 'using Pkg; Pkg.add(name="PythonCall", version="0.9.37"); Pkg.precompile()'
# JETLS language server (installed as a Pkg app into ~/.julia/bin; slow, compiles a lot)
julia -e 'using Pkg; Pkg.Apps.add(; url="https://github.com/aviatesk/JETLS.jl", rev="release")'
# Runic formatter (`runic`, `git-runic` into ~/.julia/bin)
julia -e 'using Pkg; Pkg.Apps.add("Runic")'

# verify
/opt/jax-env/bin/ipython --version
julia -e 'using PythonCall; jax = pyimport("jax"); println(jax.__version__, " ", jax.devices())'
echo SETUP_DONE
