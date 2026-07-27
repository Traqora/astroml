import { useEffect, useState } from 'react'
import { subscribeToIncomingTransactions } from '../api/loyalty'
import type { StellarTransaction } from '../lib/types'

const MAX_TRANSACTIONS = 30

export function useIncomingTransactions() {
  const [transactions, setTransactions] = useState<StellarTransaction[]>([])

  useEffect(() => {
    const unsubscribe = subscribeToIncomingTransactions((transaction) => {
      setTransactions((prev) => [transaction, ...prev].slice(0, MAX_TRANSACTIONS))
    })
    return unsubscribe
  }, [])

  return transactions
}
