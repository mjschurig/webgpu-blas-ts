/**
 * Simple HTTP server for serving WebGPU test files
 */
import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PORT = 3001;

const mimeTypes: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wgsl': 'text/plain',
};

const server = createServer(async (req, res) => {
  try {
    // Set CORS headers for cross-origin requests
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    // Handle CORS preflight
    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    let filePath = req.url === '/' ? '/test.html' : req.url;
    if (!filePath) filePath = '/test.html';
    
    // Serve from project root or test directory
    const possiblePaths = [
      join(__dirname, '..', '..', '..', 'dist', filePath.slice(1)),
      join(__dirname, filePath.slice(1)),
      join(__dirname, '..', '..', '..', filePath.slice(1)),
    ];

    let fileContent: Buffer | null = null;
    let foundPath = '';

    for (const path of possiblePaths) {
      try {
        fileContent = await readFile(path);
        foundPath = path;
        break;
      } catch {
        continue;
      }
    }

    if (!fileContent) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('File not found');
      return;
    }

    const ext = extname(foundPath);
    const contentType = mimeTypes[ext] || 'text/plain';
    
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(fileContent);
    
  } catch (error) {
    console.error('Server error:', error);
    res.writeHead(500, { 'Content-Type': 'text/plain' });
    res.end('Internal server error');
  }
});

server.listen(PORT, () => {
  console.log(`Test server running at http://localhost:${PORT}`);
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  server.close(() => {
    console.log('Test server closed');
    process.exit(0);
  });
});
