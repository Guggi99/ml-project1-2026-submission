#!/usr/bin/env bash
# Run locally what CI runs, before every push. From the repo root:
#
#     bash tools/check.sh
#
# 1. our tests (incl. doctests and the notebook-output check)  -> CI job "ours"
# 2. black formatting                                          -> CI job "ours"
# 3. the course's official grading tests, if a copy is found   -> CI job "official"
#
# The official tests are looked up in $GRADING_TESTS, then in the two places
# our guides put them (docs/github-workflow.md, or inside the course clone).
# They stay red until all six methods and run.py exist; "ours" must be green.

cd "$(dirname "$0")/.." || exit 1
status=0

echo "== 1/3 our tests"
python -m pytest -q || status=1

echo "== 2/3 formatting"
black --check . || status=1

echo "== 3/3 official grading tests"
repo="$PWD"
found=""
for dir in "$GRADING_TESTS" ../ml-course-tests/projects/project1/grading_tests \
    ../projects/project1/grading_tests; do
    if [ -n "$dir" ] && [ -f "$dir/test_project1_public.py" ]; then
        found="$dir"
        break
    fi
done
if [ -n "$found" ]; then
    (cd "$found" && python -m pytest -q --github_link "$repo" . \
        -k "not github_link_format") || echo "   (official is red: see above; not counted)"
else
    echo "   not found - skipped (set GRADING_TESTS to its folder)"
fi

if [ $status -eq 0 ]; then
    echo "== ours: GREEN - ready to push"
else
    echo "== ours: RED - fix before pushing"
fi
exit $status
