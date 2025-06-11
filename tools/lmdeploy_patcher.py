import os
import shutil
import re
from pathlib import Path
from loguru import logger
import importlib.util

class LMDeployPatcher:
    def __init__(self):
        self.lmdeploy_path = self._find_lmdeploy_path()
        self.flashattention_file = None
        self.backup_file = None
        self.target_line = "BLOCK_M = min(128, BLOCK_M)"
        self.new_line = "BLOCK_N = min(64, BLOCK_N)"
        
        if self.lmdeploy_path:
            self.flashattention_file = os.path.join(
                self.lmdeploy_path, 
                "pytorch", "kernels", "cuda", "flashattention.py"
            )
            self.backup_file = self.flashattention_file + ".backup"
            logger.info(f"Found LMDeploy path: {self.lmdeploy_path}")
            logger.info(f"Target file: {self.flashattention_file}")
        else:
            logger.error("LMDeploy installation path not found")
    
    def _find_lmdeploy_path(self):
        """Find the installation path of LMDeploy library"""
        try:
            import lmdeploy
            lmdeploy_path = os.path.dirname(lmdeploy.__file__)
            logger.info(f"Found LMDeploy via import: {lmdeploy_path}")
            return lmdeploy_path
        except ImportError:
            logger.warning("Cannot import lmdeploy directly")
        
        # Try to find from common installation paths
        possible_paths = [
            # Conda environments
            os.path.expanduser("~/anaconda3/envs/*/lib/python*/site-packages/lmdeploy"),
            os.path.expanduser("~/miniconda3/envs/*/lib/python*/site-packages/lmdeploy"),
            # System Python
            "/usr/local/lib/python*/site-packages/lmdeploy",
            "/usr/lib/python*/site-packages/lmdeploy",
            # User local installation
            os.path.expanduser("~/.local/lib/python*/site-packages/lmdeploy"),
        ]
        
        import glob
        for pattern in possible_paths:
            matches = glob.glob(pattern)
            for match in matches:
                if os.path.isdir(match) and os.path.exists(os.path.join(match, "pytorch", "kernels", "cuda", "flashattention.py")):
                    logger.info(f"Found LMDeploy path: {match}")
                    return match
        
        # Finally try using pip show
        try:
            import subprocess
            result = subprocess.run(['pip', 'show', 'lmdeploy'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Location:'):
                        location = line.split(':', 1)[1].strip()
                        lmdeploy_path = os.path.join(location, 'lmdeploy')
                        if os.path.exists(lmdeploy_path):
                            logger.info(f"Found LMDeploy via pip show: {lmdeploy_path}")
                            return lmdeploy_path
        except Exception as e:
            logger.warning(f"pip show failed: {e}")
        
        return None
    
    def _check_file_exists(self):
        """Check if the target file exists"""
        if not self.flashattention_file or not os.path.exists(self.flashattention_file):
            logger.error(f"File does not exist: {self.flashattention_file}")
            return False
        return True
    
    def _create_backup(self):
        """Create backup file"""
        if not self._check_file_exists():
            return False
        
        try:
            shutil.copy2(self.flashattention_file, self.backup_file)
            logger.info(f"Backup created: {self.backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def _find_target_line(self, content):
        """Find the position of target line (second occurrence)"""
        lines = content.split('\n')
        occurrence_count = 0
        for i, line in enumerate(lines):
            if self.target_line in line:
                occurrence_count += 1
                if occurrence_count == 2:  # Return only the second occurrence
                    return i, lines
        return -1, lines
    
    def _get_line_indentation(self, line):
        """Get the indentation of a line"""
        return len(line) - len(line.lstrip())
    
    def patch(self):
        """Apply patch"""
        if not self._check_file_exists():
            return False
        
        # Create backup
        if not self._create_backup():
            return False
        
        try:
            # Read file content
            with open(self.flashattention_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find target line
            target_line_idx, lines = self._find_target_line(content)
            if target_line_idx == -1:
                logger.error(f"Target line not found: {self.target_line}")
                return False
            
            # Check if already modified
            if target_line_idx + 1 < len(lines) and self.new_line in lines[target_line_idx + 1]:
                logger.warning("File seems to be already modified, skipping")
                return True
            
            # Get indentation of target line
            target_line = lines[target_line_idx]
            indentation = ' ' * self._get_line_indentation(target_line)
            new_line_with_indent = indentation + self.new_line
            
            # Insert new line after target line with proper indentation
            lines.insert(target_line_idx + 1, new_line_with_indent)
            
            # Write back to file
            with open(self.flashattention_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"Successfully modified file: {self.flashattention_file}")
            logger.info(f"Added after line {target_line_idx + 1}: {new_line_with_indent}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to modify file: {e}")
            # Try to restore backup
            self.restore()
            return False
    
    def restore(self):
        """Restore original file"""
        if not os.path.exists(self.backup_file):
            logger.error(f"Backup file does not exist: {self.backup_file}")
            return False
        
        try:
            shutil.copy2(self.backup_file, self.flashattention_file)
            logger.info(f"Original file restored: {self.flashattention_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore file: {e}")
            return False
    
    def check_status(self):
        """Check current status of the file"""
        if not self._check_file_exists():
            return "File does not exist"
        
        try:
            with open(self.flashattention_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            target_line_idx, lines = self._find_target_line(content)
            if target_line_idx == -1:
                return "Target line (second occurrence) not found"
            
            if target_line_idx + 1 < len(lines) and self.new_line in lines[target_line_idx + 1]:
                return "Modified"
            else:
                return "Not modified"
                
        except Exception as e:
            return f"Check failed: {e}"
    
    def clean_backup(self):
        """Clean backup file"""
        if os.path.exists(self.backup_file):
            try:
                os.remove(self.backup_file)
                logger.info(f"Backup file deleted: {self.backup_file}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete backup file: {e}")
                return False
        return True


def main():
    """Main function providing command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LMDeploy flashattention.py modification tool")
    parser.add_argument('action', choices=['patch', 'restore', 'status', 'clean'], 
                       help='Action type: patch=apply patch, restore=restore original file, status=check status, clean=clean backup')
    
    args = parser.parse_args()
    
    patcher = LMDeployPatcher()
    
    if args.action == 'patch':
        if patcher.patch():
            print("✅ Patch applied successfully")
        else:
            print("❌ Failed to apply patch")
    
    elif args.action == 'restore':
        if patcher.restore():
            print("✅ File restored successfully")
        else:
            print("❌ Failed to restore file")
    
    elif args.action == 'status':
        status = patcher.check_status()
        print(f"📋 File status: {status}")
        if os.path.exists(patcher.backup_file):
            print(f"💾 Backup file exists: {patcher.backup_file}")
    
    elif args.action == 'clean':
        if patcher.clean_backup():
            print("✅ Backup file cleaned successfully")
        else:
            print("❌ Failed to clean backup file")


if __name__ == "__main__":
    main()
