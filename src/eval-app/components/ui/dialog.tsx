"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  children: React.ReactNode
  className?: string
}

const DialogContext = React.createContext<{ onOpenChange: (open: boolean) => void } | null>(null)

function Dialog({ open, onOpenChange, children, className }: DialogProps) {
  const ref = React.useRef<HTMLDialogElement>(null)
  React.useEffect(() => {
    const el = ref.current
    if (!el) return
    if (open) {
      el.showModal()
    } else {
      el.close()
    }
  }, [open])
  const handleClose = React.useCallback(() => onOpenChange(false), [onOpenChange])
  return (
    <DialogContext.Provider value={{ onOpenChange: handleClose }}>
      <dialog
        ref={ref}
        onCancel={handleClose}
        onClick={(e) => e.target === e.currentTarget && handleClose()}
        className={cn("fixed inset-0 z-50 max-h-[90vh] w-full max-w-4xl rounded-lg border bg-background p-0 shadow-lg outline-none backdrop:bg-black/50", className)}
      >
        {children}
      </dialog>
    </DialogContext.Provider>
  )
}

function DialogHeader({ className, ...props }: React.ComponentProps<"div">) {
  return <div className={cn("flex flex-col gap-1.5 p-4 pb-2", className)} {...props} />
}

function DialogTitle({ className, ...props }: React.ComponentProps<"h2">) {
  return <h2 className={cn("text-lg font-semibold", className)} {...props} />
}

function DialogContent({ className, ...props }: React.ComponentProps<"div">) {
  return <div className={cn("overflow-y-auto p-4 pt-0 max-h-[calc(90vh-6rem)]", className)} {...props} />
}

function DialogClose({ className, children, ...props }: React.ComponentProps<"button">) {
  const ctx = React.useContext(DialogContext)
  return (
    <button
      type="button"
      onClick={() => ctx?.onOpenChange(false)}
      className={cn("rounded-md px-3 py-1.5 text-sm font-medium hover:bg-muted", className)}
      {...props}
    >
      {children ?? "Close"}
    </button>
  )
}

export { Dialog, DialogHeader, DialogTitle, DialogContent, DialogClose }
