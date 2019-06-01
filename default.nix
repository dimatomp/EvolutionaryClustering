with import <nixos-18.03> { overlays = [(self: super: {
  xgboost = super.xgboost.overrideAttrs (oldAttrs: rec {
    name = "xgboost-${version}";
    version = "0.82";

    # needs submodules
    src = self.fetchgit {
      url = "https://github.com/dmlc/xgboost";
      rev = "refs/tags/v${version}";
      sha256 = "04m48wjk3qfwgr4p4jm3r4s565bwg0pqjvn9b3zcgffy5fly2xmp";
    };
  });
})];};
let
  pythonPackageOverrides = self: super:
    let 
      buildPythonPackage = self.buildPythonPackage; 
      fetchPypi = self.fetchPypi;
    in rec {
      liac-arff = buildPythonPackage rec {
        pname = "liac-arff";
        version = "2.4.0";
        src = fetchPypi {
          inherit pname version;
          sha256 = "1xax0rdi7wijpl61v0fh5wngqxh55m18fn87cqpqkcj8s8gwvbs7";
        };
        doCheck = false;
      };

      ConfigSpace = buildPythonPackage rec {
        pname = "ConfigSpace";
        version = "0.4.10";
        src = fetchPypi {
          inherit pname version;
          sha256 = "15kxbbp84jj8pa74y3py4p8s2ax84i47jk3srbvspzh1j69h90hh";
        };
        checkInputs = with self; [ pytest ];
        propagatedBuildInputs = with self; [ numpy cython pyparsing typing ];
      };

      pyrfr = buildPythonPackage rec {
        pname = "pyrfr";
        version = "0.7.4";
        src = fetchPypi {
          inherit pname version;
          sha256 = "1r1b4iwnmccyhvcyhsjwi238n2wfn64rksaiz5g1734kim623vnh";
        };
        nativeBuildInputs = [ swig ];
      };

      smac = buildPythonPackage rec {
        pname = "smac";
        version = "0.8.0";
        src = fetchPypi {
          inherit pname version;
          sha256 = "1vp6j4w0gy61rwav5mx3nc92xss5j2fpspqim0pk14j2xjhj0ifx";
        };
        propagatedBuildInputs = with self; [ cython numpy scipy six psutil pynisher ConfigSpace scikitlearn typing pyrfr sphinx sphinx_rtd_theme joblib ];
        doCheck = false;
      };

      xgboost = super.xgboost.overrideAttrs (oldAttrs: {
        patches = [
          (substituteAll {
            src = ./lib-path-for-python.patch;
            libpath = "${pkgs.xgboost}/lib";
          })
        ];

        preInstall = "rm ../tests/python/test_dt.py ../tests/python/test_with_pandas.py";

        buildInputs = oldAttrs.buildInputs ++ (with self; [ graphviz pandas matplotlib scikitlearn pytest ]);
        doCheck = false;
      });

      matplotlib = super.matplotlib.override { enableQt = true; };

      pynisher = buildPythonPackage rec {
        pname = "pynisher";
        version = "0.5.0";
        src = fetchPypi {
          inherit pname version;
          sha256 = "0d1zjncnsdnyq8igq0s5wrj449z5d8gp1bf913hrs06ib8qpj40v";
        };
        propagatedBuildInputs = with self; [ psutil docutils ];
      };

      auto-sklearn = buildPythonPackage rec {
        pname = "auto-sklearn";
        version = "0.5.1";
        src = fetchPypi {
          inherit pname version;
          sha256 = "0n7ifbgliq3k897qz05vlbzcspcqp0kb5c0jpi3wwaqf7bbf3i1y";
        };
        propagatedBuildInputs = with self; [ nose cython numpy scipy scikitlearn xgboost lockfile joblib psutil pyyaml liac-arff pandas ConfigSpace pynisher pyrfr smac ];
        doCheck = false;
      };
    };
in
  ((python3.override { packageOverrides = pythonPackageOverrides; }).withPackages (pkgs: with pkgs; [numpy scipy matplotlib scikitlearn pyqt5 notebook pandas ipython auto-sklearn pulp]))
