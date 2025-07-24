#!/usr/bin/env python3
"""
Fix DOT file line endings for online Graphviz viewers
"""

def fix_dot_file(input_file, output_file):
    """Fix line endings and clean up DOT file."""
    print(f"🔧 Fixing DOT file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix line endings - replace Windows CRLF with Unix LF
    content = content.replace('\r\n', '\n')
    content = content.replace('\r', '\n')
    
    # Clean up any problematic characters
    content = content.replace('\x00', '')  # Remove null characters
    
    # Write with Unix line endings
    with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    
    print(f"✅ Fixed DOT file saved: {output_file}")
    print(f"📁 Ready for online Graphviz viewers!")


if __name__ == "__main__":
    fix_dot_file("scripts/model_export/WaveSourceMiniResNet_torchviz", 
                 "scripts/model_export/WaveSourceMiniResNet_fixed.dot") 