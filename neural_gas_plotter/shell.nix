{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
	python3
	python3Packages.numpy
    	python3Packages.pandas
    	python3Packages.matplotlib
	python3Packages.scikitlearn
vscodium
  ];
}
