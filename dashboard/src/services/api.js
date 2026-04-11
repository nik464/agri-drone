/**
 * API Service - Handles all communication with AgriDrone backend
 * 
 * Endpoints:
 * - POST /detect - Upload image for detection
 * - GET /health - Check API health status
 * - GET /system - Get system information
 */

import axios from 'axios'

// Configure API base URL (can be overridden with VITE_API_URL env var)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create Axios instance with defaults
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 60000 // 60 second timeout for large file uploads
})

/**
 * API Methods - Clean interface for dashboard components
 */
export const api = {
  /**
   * Send image for detection
   * @param {File} file - Image file to process
   * @param {Boolean} useMockModel - Use mock detection for demo (default: false)
   * @returns {Promise} Detection results with bounding boxes
   */
  detectObjects: async (file, useMockModel = false) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('use_mock', useMockModel)

    try {
      const response = await apiClient.post('/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      return response.data
    } catch (error) {
      throw {
        message: error.response?.data?.message || error.message,
        status: error.response?.status,
        error
      }
    }
  },

  /**
   * Check if API is healthy and responsive
   * @returns {Promise} Health status (ok/disconnected)
   */
  getHealth: async () => {
    try {
      const response = await apiClient.get('/health')
      return response.data
    } catch (error) {
      return { status: 'disconnected', error: error.message }
    }
  },

  /**
   * Get system information and configuration
   * @returns {Promise} System info (version, device, etc)
   */
  getSystemInfo: async () => {
    try {
      const response = await apiClient.get('/system')
      return response.data
    } catch (error) {
      return { error: error.message }
    }
  }
}

export default apiClient

