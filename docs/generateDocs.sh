#!/bin/bash
docsDir=docsTmpFiles #change this to your package name

#get the folder this script is in (so the script can be run safely from any location and still work)
scriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $scriptDir                    #go to the directory the script is in

rm -rf $docsDir                  #remove old docs
make html                        #gen new docs

cp -r _build/html ${docsDir}     #copy build output to docs dir
cp -r src/autoGen ${docsDir}/src #copy video files into docs

rm -rf _build #remove build files

#view in firefox
#firefox ${docsDir}/src/index.html &

#use rsync to copy to the gpu sever for hosting
#rsync -avzh ./${docsDir} devops@10.48.83.14:/home/devops/projects/api_docs/docs
#password is "myrobotrocks"  you probably want to use ssh-copy-id so that you don't have to type the password each time you update your docs
# rsync -avzh docsTmpFiles/* devops@10.48.83.14:/home/devops/projects/api_docs/docs/grip_docs