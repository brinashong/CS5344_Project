if [[ -z "$IN_NIX_SHELL" ]]; then
  return 0
elif [[ -n "$IN_NIX_SHELL" && "$IN_NIX_SHELL" == "impure" ]]; then
  micromamba deactivate
fi
