"use client";

import { useState, useMemo } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import WatchForm from "@/features/watch/components/watch-form";

import { useToast } from "@/hooks/use-toast";

import useSignalStore from "@/app/store/signalStore";

import { type Config } from "@/types";

export default function Watch() {
  const { signals } = useSignalStore();
  const [editMode, setEditMode] = useState(false);
  const [data, setData] = useState<Config>();
  const [openDialog, setOpenDialog] = useState<string | null>(null);

  const showEditMode = useMemo(() => {
    return editMode || !data;
  }, [editMode, data]);

  const handleDialogToggle = (dialog: string | null) => {
    setOpenDialog(dialog);
  };

  return (
    <>
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Bybit</CardTitle>
          <CardDescription>
            {showEditMode
              ? "Add your Bybit account"
              : "Bybit account configured"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Dialog
              open={openDialog === "bybit"}
              onOpenChange={() =>
                handleDialogToggle(openDialog ? null : "bybit")
              }
            >
              <DialogTrigger className="flex-1 rounded-md border px-4 py-2 font-mono text-sm shadow-sm">
                Bybit
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Your Bybit API Keys</DialogTitle>
                  <DialogDescription>
                    Consectetur deserunt cillum officia in reprehenderit irure
                    amet id anim nostrud irure nisi proident consectetur
                    reprehenderit.
                  </DialogDescription>
                </DialogHeader>
                <WatchForm
                  exchange="bybit"
                  onSubmitAction={() =>
                    handleDialogToggle(openDialog ? null : "bybit")
                  }
                  onCancelAction={() => handleDialogToggle(null)}
                />
              </DialogContent>
            </Dialog>
            <Dialog
              open={openDialog === "mexc"}
              onOpenChange={() =>
                handleDialogToggle(openDialog ? null : "mexc")
              }
            >
              <DialogTrigger className="flex-1 rounded-md border px-4 py-2 font-mono text-sm shadow-sm">
                Mexc
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Your Mexc API Keys</DialogTitle>
                  <DialogDescription>
                    Consectetur deserunt cillum officia in reprehenderit irure
                    amet id anim nostrud irure nisi proident consectetur
                    reprehenderit.
                  </DialogDescription>
                </DialogHeader>
                <WatchForm
                  exchange="mexc"
                  onSubmitAction={() =>
                    handleDialogToggle(openDialog ? null : "mexc")
                  }
                  onCancelAction={() => handleDialogToggle(null)}
                />
              </DialogContent>
            </Dialog>
          </div>
        </CardContent>
      </Card>

      <Sheet>
        <SheetTrigger className="text-xs text-center w-full mt-4 hover:underline hover:cursor-pointer">
          View Recent Signals
        </SheetTrigger>
        <SheetContent>
          <SheetHeader>
            <SheetTitle className="mb-6">Recent Signals</SheetTitle>
            <div>
              {signals.map((signal, index) => (
                <Card key={index} className="mb-4">
                  <CardHeader>
                    <CardTitle>Signal sent to PTN </CardTitle>
                    <CardDescription>
                      {signal.orderType}:{signal.trade_pair.trade_pair} Lvg:{" "}
                      {signal.leverage}
                    </CardDescription>
                  </CardHeader>
                </Card>
              ))}
            </div>
          </SheetHeader>
        </SheetContent>
      </Sheet>
    </>
  );
}
