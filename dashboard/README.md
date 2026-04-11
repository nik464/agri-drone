# AgriDrone AI Dashboard

Premium, production-grade web dashboard for AgriDrone detection system powered by React, Vite, and Tailwind CSS.

## Features

✨ **Modern Design**
- Dark SaaS-style theme with glassmorphism effects
- Smooth Framer Motion animations
- Responsive layout with Tailwind CSS
- Neon blue/purple accent colors

🎯 **Core Features**
- Drag & drop image upload with live preview
- Real-time AI detection with bounding boxes
- Sortable detection table with confidence scores
- Statistics cards (total detections, avg confidence, processing time)
- JSON & CSV export functionality
- Detection history tracking
- API status indicator
- Mock vs Real model toggle

🔌 **Integration**
- FastAPI backend integration
- Multipart image upload via Axios
- Health check monitoring
- Error handling & loading states
- Graceful offline mode

## Tech Stack

- **React 18** - UI framework
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Axios** - HTTP client
- **Lucide React** - Modern icon set

## Installation

### Prerequisites
- Node.js 16+ and npm/yarn installed
- AgriDrone FastAPI backend running on `localhost:8000`

### Setup

```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Start development server
npm run dev
```

The dashboard will open at `http://localhost:5173`

## Environment Variables

Create a `.env` file:

```env
# API Configuration
VITE_API_URL=http://localhost:8000
```

## Building for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── Navbar.jsx          # Top navigation with API status
│   ├── Sidebar.jsx         # Left sidebar navigation
│   ├── UploadBox.jsx       # Drag & drop upload zone
│   ├── ResultViewer.jsx    # Image with bounding boxes
│   ├── DetectionTable.jsx  # Sortable detections table
│   └── StatsCards.jsx      # Statistics display
├── services/
│   └── api.js             # API client with Axios
├── App.jsx                 # Main application component
├── main.jsx                # Entry point
└── index.css              # Global styles & animations
```

## API Requirements

The FastAPI backend should provide:

### POST /detect
Send image for detection

**Request:**
```
Content-Type: multipart/form-data
file: <image_file>
use_mock: boolean
```

**Response:**
```json
{
  "detections": [
    {
      "id": "det_1",
      "class_name": "weed",
      "confidence": 0.95,
      "x1": 100,
      "y1": 150,
      "x2": 200,
      "y2": 250
    }
  ],
  "image_base64": "data:image/jpeg;base64,..."
}
```

### GET /health
API health check

**Response:**
```json
{
  "status": "ok"
}
```

## Design System

### Colors
- **Dark Theme**: `#0f172a` (bg-dark)
- **Accent Blue**: `#0ea5e9` (neon-blue)
- **Accent Purple**: `#a855f7` (neon-purple)
- **Accent Cyan**: `#06b6d4` (neon-cyan)

### Components
- **Glassmorphism**: Blurred glass effect with transparency
- **Animations**: Smooth transitions and interactive effects
- **Responsive**: Works on desktop, tablet, mobile

## Features in Detail

### Upload Section
- Drag & drop or click to upload
- Image preview with dimensions
- Mock model toggle for testing
- Real-time validation

### Detection Results
- Processed image with bounding boxes
- Color-coded by class (weed, disease, pest, anomaly)
- Confidence scores for each detection
- Processing time display

### Detection Table
- Sortable columns (class, confidence, bbox, area)
- Color-coded class tags
- Animated confidence bars
- Responsive horizontal scroll

### Statistics
- Total detections count
- Average confidence percentage
- Processing time in milliseconds
- Hover animations

### Export
- Download as JSON with full metadata
- Download as CSV for spreadsheet analysis
- Timestamp included in exports

### History
- Last 10 detection sessions
- Quick access to previous runs
- Detection count summary per session

## Keyboard Shortcuts

- `Ctrl/Cmd + U` - Goto Upload (coming soon)
- `Ctrl/Cmd + D` - Goto Dashboard (coming soon)
- `Ctrl/Cmd + H` - Goto History (coming soon)

## Performance

- Vite HMR for instant dev updates
- Lazy loading components
- Optimized re-renders with React.memo
- Canvas-based visualization
- Efficient state management

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari 14+, Chrome Android)

## Troubleshooting

### API Connection Failed
1. Ensure FastAPI backend is running on `http://localhost:8000`
2. Check VITE_API_URL in .env
3. Check browser console for CORS errors
4. Verify backend has CORS enabled for http://localhost:5173

### Image Upload Issues
1. Check file size (max 50MB)
2. Verify image format (PNG, JPG, GIF)
3. Check browser console for detailed errors

### Styling Issues
1. Clear node_modules: `rm -rf node_modules && npm install`
2. Rebuild Tailwind: `npm run dev`
3. Check Tailwind config is correct

## Development Tips

- Use React DevTools extension for debugging
- Check Network tab in DevTools for API calls
- Use `VITE_DEBUG=true` for verbose logging (coming soon)
- Mobile preview: `localhost:5173` on your phone (same network)

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- Code follows React best practices
- Components are properly typed
- Animations are smooth (60fps)
- Mobile responsive
- Accessibility compliance

## Support

For issues or questions:
1. Check troubleshooting section
2. Review browser console
3. Check FastAPI backend logs
4. Create an issue with error details and screenshots

---

**Built for AgriDrone Research Project**
Research prototype for site-specific crop protection using aerial imagery and AI detection.
