require(knitr) # required for knitting from rmd to md
require(markdown) # required for md to html
#knit('data_preprocess.Rmd', 'test.md') # creates md file
filename = 'feature_select'
knit(paste(filename, ".Rmd", sep = ""), 'test.md') # creates md file
markdownToHTML('test.md', paste(filename, ".html", sep = "")) # creates html file
browseURL(paste('file://', file.path(getwd(), paste(filename, ".html", sep = "")), sep='')) # open file in browser
