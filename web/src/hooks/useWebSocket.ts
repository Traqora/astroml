/**
 * WebSocket Hook for Real-time Data
 * 
 * Provides a React hook for managing WebSocket connections with automatic reconnection,
 * error handling, and cleanup.
 */

import { useEffect, useRef, useState, useCallback } from 'react'

interface WebSocketHookOptions {
  url: string
  onMessage?: (data: any) => void
  onError?: (error: Event) => void
  onOpen?: (event: Event) => void
  onClose?: (event: Event) => void
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

interface WebSocketHookReturn {
  isConnected: boolean
  lastMessage: any
  error: Event | null
  sendMessage: (data: any) => void
  connect: () => void
  disconnect: () => void
}

export function useWebSocket({
  url,
  onMessage,
  onError,
  onOpen,
  onClose,
  reconnectInterval = 3000,
  maxReconnectAttempts = 5,
}: WebSocketHookOptions): WebSocketHookReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<any>(null)
  const [error, setError] = useState<Event | null>(null)
  
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = (event) => {
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0
        onOpen?.(event)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setLastMessage(data)
          onMessage?.(data)
        } catch (err) {
          // If not JSON, pass as-is
          setLastMessage(event.data)
          onMessage?.(event.data)
        }
      }

      ws.onerror = (event) => {
        setError(event)
        onError?.(event)
      }

      ws.onclose = (event) => {
        setIsConnected(false)
        onClose?.(event)

        // Attempt reconnection if not closed intentionally
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }
    } catch (err) {
      setError(err as Event)
      onError?.(err as Event)
    }
  }, [url, onMessage, onError, onOpen, onClose, reconnectInterval, maxReconnectAttempts])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setIsConnected(false)
    reconnectAttemptsRef.current = 0
  }, [])

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        const message = typeof data === 'string' ? data : JSON.stringify(data)
        wsRef.current.send(message)
      } catch (err) {
        setError(err as Event)
        onError?.(err as Event)
      }
    }
  }, [onError])

  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
    connect,
    disconnect,
  }
}

/**
 * Hook for subscribing to real-time transaction updates
 */
export function useTransactionUpdates(onTransaction: (transaction: any) => void) {
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  const wsUrl = (import.meta.env.VITE_WS_URL as string | undefined)
    || `${apiBase.replace(/^http/, 'ws')}/api/v1/ws/transactions`

  return useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      if (data.type === 'transaction') {
        onTransaction(data.data)
      }
    },
    onError: (error) => {
      console.error('WebSocket error:', error)
    },
  })
}

/**
 * Hook for subscribing to real-time fraud alerts
 */
export function useFraudAlerts(onAlert: (alert: any) => void) {
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
  const wsUrl = (import.meta.env.VITE_WS_URL as string | undefined)
    || `${apiBase.replace(/^http/, 'ws')}/api/v1/ws/alerts`

  return useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      if (data.type === 'fraud_alert') {
        onAlert(data.data)
      }
    },
    onError: (error) => {
      console.error('WebSocket error:', error)
    },
  })
}
