#!/usr/bin/env python3
import sys
import subprocess
import os
import stat # 导入 stat 模块以使用权限常量

def main():
    """
    一个简单的 Python 包装器，用于调用 merge_model.sh 脚本。
    它会先确保 Shell 脚本有执行权限，然后再调用它，并将所有命令行参数传递过去。
    """
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建要调用的 Shell 脚本的完整路径
    shell_script_path = os.path.join(script_dir, "merge_model.sh")

    # 检查 Shell 脚本是否存在
    if not os.path.exists(shell_script_path):
        print(f"错误: Shell 脚本未找到，请确保 'merge_model.sh' 与此 Python 脚本在同一目录下。", file=sys.stderr)
        print(f"期望路径: {shell_script_path}", file=sys.stderr)
        sys.exit(1)

    # --- 新增：在执行前确保脚本有执行权限 ---
    try:
        print(f"--- Python Wrapper: 正在检查并设置脚本权限 ---")
        # 获取当前文件的权限
        current_permissions = stat.S_IMODE(os.stat(shell_script_path).st_mode)
        
        # 添加用户、组和其他用户的执行权限 (等同于 chmod a+x)
        # stat.S_IXUSR: 用户执行权限
        # stat.S_IXGRP: 组执行权限
        # stat.S_IXOTH: 其他人执行权限
        new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        
        if new_permissions != current_permissions:
            os.chmod(shell_script_path, new_permissions)
            print(f"--- Python Wrapper: 已为 '{shell_script_path}' 添加执行权限 ---")
        else:
            print(f"--- Python Wrapper: 脚本已有执行权限，无需更改 ---")

    except Exception as e:
        print(f"错误: 尝试为脚本设置权限时失败: {e}", file=sys.stderr)
        print("请检查当前用户是否有权限修改该文件（是否为文件所有者或有足够权限）。", file=sys.stderr)
        sys.exit(1)
    # --- 权限设置结束 ---

    # 准备要执行的命令
    command = [shell_script_path] + sys.argv[1:]

    print(f"--- Python Wrapper: 正在调用 Shell 脚本 ---")
    print(f"--- 命令: {' '.join(command)} ---")
    print("-" * 40)

    # 使用 subprocess.run 来执行 Shell 脚本
    try:
        # 使用 Popen 进行流式输出，对调试更友好
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # 实时读取并打印输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, command)

        print("-" * 40)
        print(f"--- Python Wrapper: Shell 脚本执行成功 ---")

    except FileNotFoundError:
        # 这个错误理论上不会再因为权限问题发生，但保留以防万一
        print(f"错误: 命令 '{command[0]}' 未找到或不可执行。", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # Shell 脚本自身执行失败（例如 exit 1）
        print("-" * 40, file=sys.stderr)
        print(f"--- Python Wrapper: Shell 脚本执行失败，退出码: {e.returncode} ---", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"--- Python Wrapper: 发生未知错误: {e} ---", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
