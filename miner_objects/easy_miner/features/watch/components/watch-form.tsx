"use client";

import { useEffect, MouseEvent } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { toast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { type Config } from '@/types'
import {Switch} from "@/components/ui/switch";
import {Label} from "@/components/ui/label";

const FormSchema = z.object({
  apiKey: z.string().min(1, {
    message: "API Key must be at least 1 characters.",
  }),
  secret: z.string().min(1, {
    message: "API Secret must be at least 1 characters.",
  }),
  demo: z.boolean(),
});

type OnSubmit = (data: z.infer<typeof FormSchema>) => void;

interface UploadFormProps {
  onSubmitAction: OnSubmit;
  onCancelAction: () => void;
  data?: Config;
}

export default function WatchForm({
  onSubmitAction,
  onCancelAction,
  data
}: UploadFormProps) {
  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      apiKey: "",
      secret: "",
      demo: true,
    },
  });

  useEffect(() => {
    if (data) {
      form.reset({
        apiKey: data.apiKey,
        secret: data.secret,
        demo: data.demo,
      });
    }
  }, [data, form]);

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

  function onCancel(e: MouseEvent<HTMLElement>): void {
    e.preventDefault();

    onCancelAction();
  }

  return (
    <Form {...form}>
      <form
        className="grid w-full items-center gap-4"
        onSubmit={form.handleSubmit(handleSubmit)}
      >
        <FormField
          control={form.control}
          name="apiKey"
          render={({field}) => (
            <FormItem>
              <FormLabel>API Key</FormLabel>
              <FormControl>
                <Input placeholder="Enter your API Key" {...field} />
              </FormControl>
              <FormMessage/>
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="secret"
          render={({field}) => (
            <FormItem>
              <FormLabel>API Secret</FormLabel>
              <FormControl>
                <Input
                  placeholder="Enter your API Secret"
                  type="password"
                  {...field}
                />
              </FormControl>
              <FormMessage/>
            </FormItem>
          )}
        />
        <div className="flex justify-end gap-4 w-full my-4">
          <FormField
            control={form.control}
            name="demo"
            render={({field}) => (
              <FormItem>
                <FormControl>
                  <div className="flex justify-end items-center space-x-2">
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                    <Label htmlFor="demo">Demo Account</Label>
                  </div>
                </FormControl>
                <FormMessage/>
              </FormItem>
            )}
          />
        </div>
        <div className='flex w-full gap-8'>
          <Button className='w-full' variant="secondary" onClick={onCancel}>Cancel</Button>
          <Button className='w-full' type="submit">Save</Button>
        </div>

      </form>
    </Form>
);
}
