#!/bin/bash

clear

source ~/.bash_profile

# Configure this variable to the alias name for your python interpreter
ALIAS=deeplearn


PYTHON_INTERPRETER=$(alias "$ALIAS" | sed -E "s/alias $ALIAS='(.*)'/\1/")

echo
echo ":: AI Suite ::"
echo

# AI file must be named env.py at the same level as this script
$PYTHON_INTERPRETER ./env.py "$@"

echo
echo ":: Program succeeded with exit status '0' ::"
echo