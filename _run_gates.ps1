$py = 'C:\Users\MENDOS\AppData\Local\Programs\Python312\python.exe'
Set-Location 'c:\Users\MENDOS\Desktop\astroml'

& $py -m interrogate astroml api tests *> '_ci_interrogate2.txt'
"interrogate_exit=$LASTEXITCODE" | Out-File -Append '_ci_interrogate2.txt'

& $py -m mypy astroml/tracking/model_registry.py *> '_ci_mypy2.txt'
"mypy_exit=$LASTEXITCODE" | Out-File -Append '_ci_mypy2.txt'
