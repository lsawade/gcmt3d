# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

GH_PAGES_SOURCES = docs Makefile

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


gh-pages:
	# This line is necessary because __pycache__ files are not deleted
	# when the branch switches and therefore gcmt3d stays a dirtree in the
	# gh-pages branch
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	git checkout gh-pages
	rm -rf build _sources _static _modules _images chapters
	git checkout master $(GH_PAGES_SOURCES)
	git reset HEAD
	make html
	mv -fv docs/build/html/* ./
	rm -rf $(GH_PAGES_SOURCES) build
	git add -A
	git ci -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`" && git push origin gh-pages ; git checkout master
