import { useState, useCallback } from 'react'
import { Upload, FileImage, Loader2, CheckCircle2 } from 'lucide-react'
import { Modal } from '@/components/shared/Modal'
import { Button } from '@/components/ui/button'
import { importExternalDataset } from '@/api'
import { cn } from '@/lib/utils'

interface ImportDatasetModalProps {
  open: boolean
  onClose: () => void
}

export function ImportDatasetModal({ open, onClose }: ImportDatasetModalProps) {
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [importing, setImporting] = useState(false)
  const [done, setDone] = useState(false)
  const [message, setMessage] = useState('')

  const handleFile = async (f: File) => {
    setFile(f)
    setImporting(true)
    const result = await importExternalDataset(f)
    setImporting(false)
    setDone(result.success)
    setMessage(result.message)
  }

  const reset = useCallback(() => {
    setFile(null)
    setImporting(false)
    setDone(false)
    setMessage('')
  }, [])

  const handleClose = () => {
    reset()
    onClose()
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Import External Dataset"
      description="Secondary workflow — import GeoTIFF from local storage or external source"
    >
      {done ? (
        <div className="text-center py-4">
          <CheckCircle2 className="h-10 w-10 text-success mx-auto mb-3" />
          <p className="text-sm text-foreground mb-1">Import queued</p>
          <p className="text-xs text-muted">{message}</p>
          <Button className="mt-6" onClick={handleClose}>Done</Button>
        </div>
      ) : (
        <>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              const f = e.dataTransfer.files[0]
              if (f) handleFile(f)
            }}
            className={cn(
              'rounded-xl border-2 border-dashed p-10 text-center transition-all duration-200',
              dragging ? 'border-primary bg-primary/5' : 'border-border hover:border-border-subtle'
            )}
          >
            {importing ? (
              <Loader2 className="h-8 w-8 text-primary mx-auto animate-spin" />
            ) : (
              <>
                <Upload className="h-8 w-8 text-muted mx-auto mb-3" />
                <p className="text-sm text-foreground mb-1">Drag & drop GeoTIFF here</p>
                <p className="text-xs text-muted">Supports .tif, .tiff, .cog — max 10 GB</p>
              </>
            )}
          </div>
          <div className="flex items-center gap-3 mt-4">
            <div className="flex-1 h-px bg-border" />
            <span className="text-xs text-muted-dim">or</span>
            <div className="flex-1 h-px bg-border" />
          </div>
          <label className="mt-4 block">
            <input
              type="file"
              accept=".tif,.tiff,.geotiff"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
            />
            <Button variant="secondary" className="w-full cursor-pointer">
              <FileImage className="h-4 w-4" /> Browse GeoTIFF
            </Button>
          </label>
          {file && !importing && (
            <p className="text-xs text-muted mt-3 font-mono truncate">{file.name}</p>
          )}
        </>
      )}
    </Modal>
  )
}
