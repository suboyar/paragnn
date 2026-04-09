;;; Directory Local Variables            -*- no-byte-compile: t -*-
;;; For more information see (info "(emacs) Directory Variables")

((asm-mode . ((asm-comment-char . 35)))
 (c-mode . ((eval . (progn
                      (setq c-macro-names-with-semicolon
                            (append c-macro-names-with-semicolon '("PRAGMA_UNROLL")))
                      (c-make-macro-with-semi-re))))))
