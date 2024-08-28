import argostranslate.package

# 更新包索引
argostranslate.package.update_package_index()

# 获取可用的翻译包
available_packages = argostranslate.package.get_available_packages()

# 找到并安装中英文翻译包
for package in available_packages:
    if package.from_code == "zh" and package.to_code == "en":
        argostranslate.package.install_from_path(package.download())
        break  # 找到后立即安装，可以根据实际需求修改

# 获取已安装的翻译包
installed_packages = argostranslate.package.get_installed_packages()

print("已安装的翻译包信息：")
for package in installed_packages:
    print(f"From: {package.from_code}, To: {package.to_code}")

print("翻译包安装完成！")