CarpeDiem interactive data browser, version 1.1

To launch this in your browser, you would need a web server to host the html and js files.
Unfortunately, directly opening the `index.html` in browser will not work.

One of the easiest ways to launch a web server is from the command line with python:
1. Open Terminal or Command line
2. Navigate to the directory with the data browser
3. Run `python -mhttp.server 8000` (or any other suitable port instead of 8000)
4. Open browser to http://localhost:8000

To make this browser, we used:
* d3.js with license available in d3.js_LICENSE.txt
* pako with license available in pako_LICENSE.txt

Data browser license is availabe in LICENSE.txt