with import <nixpkgs> {};
python3.withPackages (pkgs: with pkgs; [numpy scipy (matplotlib.override {enableQt = true;})])
