'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

interface NavbarProps {
  title?: string
}

const navLinks = [
  { href: '/', label: 'Home' },
  { href: '/evaluation', label: 'Evaluation Results' },
  { href: '/run-evaluation', label: 'Run Evaluation' },
  { href: '/chat', label: 'Chat' },
  { href: '/performance-monitor', label: 'Performance Monitor' }
]

const Navbar: React.FC<NavbarProps> = ({ title = 'Basher' }) => {
  const pathname = usePathname()

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60">
      <nav className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <Link
          href="/"
          className="flex items-center gap-2 font-semibold text-foreground hover:text-foreground/80 transition-colors"
        >
          {title}
        </Link>
        <ul className="flex items-center gap-1 sm:gap-2">
          {navLinks.map(({ href, label }) => {
            const isActive = pathname === href || (href !== '/' && pathname.startsWith(href))
            return (
              <li key={href}>
                <Link
                  href={href}
                  className={`inline-flex items-center justify-center rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                  }`}
                >
                  {label}
                </Link>
              </li>
            )
          })}
        </ul>
        <ul className="flex items-center gap-1 sm:gap-2">
        <li>
        <Link href="http://localhost:6333/dashboard"
          className="inline-flex items-center justify-center rounded-md px-3 py-2 text-sm font-medium transition-colors text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          target="_blank"
          rel="noopener noreferrer"
        >
          Qdrant Dashboard
        </Link>
        </li>
        <li>
        <Link href="http://localhost:8000/docs"
          className="inline-flex items-center justify-center rounded-md px-3 py-2 text-sm font-medium transition-colors text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          target="_blank"
          rel="noopener noreferrer"
        >
          API docs
        </Link>
        </li>
        </ul>
      </nav>
    </header>
  )
}

export default Navbar
