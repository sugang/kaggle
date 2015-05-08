require(knitr) # required for knitting from rmd to md
require(markdown) # required for md to html
knit('sup_fig.Rmd', 'test.md') # creates md file
markdownToHTML('test.md', 'test.html') # creates html file
browseURL(paste('file://', file.path(getwd(),'test.html'), sep='')) # open file in browser
