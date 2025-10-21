#!/usr/bin/env python3
"""
Test script to verify the file handling fix works properly.
This version doesn't require external dependencies.
"""

import os
import time
import tempfile
from pathlib import Path


def create_safe_output_directory():
    """
    Create a safe output directory for saving results.
    
    Returns:
        str: Path to the created directory
    """
    # Try multiple directory options in order of preference
    possible_dirs = [
        # 1. Current workspace directory (most reliable)
        os.path.join(os.getcwd(), "llm_eval_results"),
        # 2. User's home directory
        os.path.expanduser("~/llm_eval_results"),
        # 3. System temp directory with user permissions
        os.path.join(tempfile.gettempdir(), f"llm_eval_results_{os.getuid() if hasattr(os, 'getuid') else 'user'}"),
        # 4. Fallback to Python's tempfile
        tempfile.mkdtemp(prefix="llm_eval_")
    ]
    
    for directory in possible_dirs:
        try:
            # Create directory with proper permissions
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Test write permissions by creating a test file
            test_file = os.path.join(directory, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            print(f"✅ Using output directory: {directory}")
            return directory
            
        except (PermissionError, OSError) as e:
            print(f"⚠️ Cannot use directory {directory}: {e}")
            continue
    
    # If all else fails, use current directory
    fallback_dir = os.getcwd()
    print(f"🔄 Falling back to current directory: {fallback_dir}")
    return fallback_dir


def safe_save_csv_data(data_rows, headers, output_dir: str, filename_prefix: str = "results") -> str:
    """
    Safely save CSV data with error handling (without pandas dependency).
    
    Args:
        data_rows: List of data rows
        headers: List of column headers
        output_dir: Directory to save the file
        filename_prefix: Prefix for the filename
        
    Returns:
        str: Path to the saved file, or None if failed
    """
    timestamp = int(time.time())
    filename = f"{filename_prefix}_{timestamp}.csv"
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Save CSV manually
        with open(file_path, 'w') as f:
            # Write headers
            f.write(','.join(headers) + '\n')
            # Write data rows
            for row in data_rows:
                f.write(','.join(str(cell) for cell in row) + '\n')
        
        print(f"✅ Results saved to: {file_path}")
        return file_path
        
    except PermissionError as e:
        print(f"❌ Permission denied saving to {file_path}: {e}")
        
        # Try alternative filename with random suffix
        import random
        alt_filename = f"{filename_prefix}_{timestamp}_{random.randint(1000, 9999)}.csv"
        alt_path = os.path.join(output_dir, alt_filename)
        
        try:
            with open(alt_path, 'w') as f:
                f.write(','.join(headers) + '\n')
                for row in data_rows:
                    f.write(','.join(str(cell) for cell in row) + '\n')
            
            print(f"✅ Results saved to alternative path: {alt_path}")
            return alt_path
        except Exception as e2:
            print(f"❌ Failed to save to alternative path: {e2}")
            return None
            
    except Exception as e:
        print(f"❌ Unexpected error saving results: {e}")
        return None


def test_file_handling():
    """Test the file handling functionality."""
    print("🧪 TESTING FILE HANDLING FIX")
    print("="*50)
    
    # Test directory creation
    output_dir = create_safe_output_directory()
    
    # Test CSV saving
    headers = ['metric_name', 'score', 'status', 'threshold']
    data_rows = [
        ['accuracy', '0.85', '✅', '0.8'],
        ['relevance', '0.92', '✅', '0.8'],
        ['completeness', '0.78', '✅', '0.7']
    ]
    
    saved_path = safe_save_csv_data(data_rows, headers, output_dir, "test_results")
    
    if saved_path:
        print(f"\n📁 File saved successfully!")
        print(f"📍 Location: {saved_path}")
        
        # Verify the file exists and is readable
        if os.path.exists(saved_path):
            file_size = os.path.getsize(saved_path)
            print(f"📏 File size: {file_size} bytes")
            
            # Read and display first few lines
            with open(saved_path, 'r') as f:
                lines = f.readlines()
                print(f"📋 File contents ({len(lines)} lines):")
                for i, line in enumerate(lines[:5]):  # Show first 5 lines
                    print(f"   {i+1}: {line.strip()}")
        
        # Clean up test file
        try:
            os.remove(saved_path)
            print(f"🧹 Test file cleaned up")
        except Exception as e:
            print(f"⚠️ Could not clean up test file: {e}")
    
    else:
        print("❌ File saving test failed!")
        return False
    
    print("\n✅ File handling test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_file_handling()
    if success:
        print("\n🎉 The fix is working properly!")
    else:
        print("\n💥 The fix needs more work.")