#!/usr/bin/env sh

pandoc -s -t html5 --template template.html5 -o public/index.html index.md
pandoc -s -t html5 --template template.html5 -o public/introduction.html introduction.md
pandoc -s -t html5 --template template.html5 -o public/description.html description.md
pandoc -s -t html5 --template template.html5 -o public/requirements.html requirements.md
pandoc -s -t html5 --template template.html5 -o public/test_plan.html test_plan.md
pandoc -s -t html5 --template template.html5 -o public/continuous_integration.html continuous_integration.md
pandoc -s -t html5 --template template.html5 -o public/references.html references.md
