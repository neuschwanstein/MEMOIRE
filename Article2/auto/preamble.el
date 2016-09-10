(TeX-add-style-hook
 "preamble"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8")))
   (TeX-run-style-hooks
    "amsmath"
    "amsthm"
    "amsfonts"
    "graphicx"
    "bm"
    "xparse"
    "inputenc"
    "mathrsfs")
   (TeX-add-symbols
    '("Thierry" 1)
    '("Erick" 1)
    '("comment" 1)
    '("maximizeEquation" 1)
    '("minimizeEquation" 1)
    '("figref" 1)
    "ie"
    "eg"
    "iid"
    "ts"
    "iso"
    "dd"
    "real"
    "normal"
    "trueRisk"
    "uInv"
    "qHat"
    "qStar"
    "xMax"
    "grad"
    "E"
    "EU"
    "pp"
    "subsetsim"
    "rf"
    "starsection"
    "rcurs")
   (LaTeX-add-environments
    "prop"
    "thm"
    "coro"
    "claim"
    "lemma"
    "assumption"))
 :latex)

