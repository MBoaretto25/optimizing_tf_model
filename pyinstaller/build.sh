rm -rf ./build/ ./dist/
echo "Beginning build"
pyinstaller -c ../run.py \
    --additional-hooks-dir ./ \
    --onefile \
    --key="key_da_fiscal_super_secreta"
