"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { toast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";

const FormSchema = z.object({
  testnet: z.boolean(),
  hotkey: z.string(),
  wallet: z.string(),
  debug: z.boolean(),
});

type OnSubmit = (data: z.infer<typeof FormSchema>) => void;

interface MinerFormProps {
  onSubmitAction: OnSubmit;
  data?: { apiKey: string; secret: string };
}

export default function MinerForm({ onSubmitAction }: MinerFormProps) {
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      testnet: true,
      debug: false,
      hotkey: "default",
      wallet: "",
    },
  });
  //
  // useEffect(() => {
  //   if (data) {
  //     form.reset({
  //       apiKey: data.apiKey,
  //       secret: data.secret,
  //     });
  //   }
  // }, [data, form]);

  function handleSubmit(data: z.infer<typeof FormSchema>) {
    toast({
      title: "You submitted the following values:",
      description: (
        <pre className="mt-2 w-[340px] rounded-md bg-slate-950 p-4">
          <code className="text-white">{JSON.stringify(data, null, 2)}</code>
        </pre>
      ),
    });

    onSubmitAction(data);
  }

  return (
    <Form {...form}>
      <form
        className="grid w-full items-center gap-4"
        onSubmit={form.handleSubmit(handleSubmit)}
      >
        <div className="flex gap-4 w-full">
          <FormField
            control={form.control}
            name="wallet"
            render={({ field }) => (
              <FormItem className="w-full">
                <FormLabel>Wallet Name</FormLabel>
                <FormControl>
                  <Input placeholder="Enter your Wallet Name" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="hotkey"
            render={({ field }) => (
              <FormItem className="w-full">
                <FormLabel>Wallet Hotkey</FormLabel>
                <FormControl>
                  <Input placeholder="Enter your Wallet Hokey" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <div className="flex justify-end gap-4 w-full my-4">
          <FormField
            control={form.control}
            name="debug"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <div className="flex justify-end items-center space-x-2">
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                    <Label htmlFor="debug">Debug</Label>
                  </div>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="testnet"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <div className="flex justify-end items-center space-x-2">
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                    <Label htmlFor="testnet">Testnet</Label>
                  </div>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <Button className="w-full" type="submit">
          Start
        </Button>
      </form>
    </Form>
  );
}
