# Show nix

if [[ -z "$IN_NIX_SHELL" ]]; then
  nix develop -c $SHELL
elif [[ -n "$IN_NIX_SHELL" && "$IN_NIX_SHELL" == "impure" ]]; then
  # Initialize Micromamba
  eval "$(micromamba shell hook --shell zsh)"

  # Check if environment exists, if not, create and install from requirements.txt
  if ! micromamba env list | grep -q "cs5344"; then
    micromamba create -y -f environment.yml
  fi
  micromamba activate cs5344
fi
